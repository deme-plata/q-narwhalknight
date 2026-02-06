/// High-performance KV store abstraction with RocksDB implementation (Linux/macOS)
/// Optimized for DagKnight/Narwhal/Bullshark access patterns
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::{collections::HashMap, path::Path, sync::Arc};
use tracing::{debug, error, info, warn};

// RocksDB imports - Linux/macOS only
#[cfg(not(target_os = "windows"))]
use q_quantum_rng::{QRNGConfig, QuantumRNG, QuantumRandomness};
#[cfg(not(target_os = "windows"))]
use q_types::Phase;
#[cfg(not(target_os = "windows"))]
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};

#[cfg(not(target_os = "windows"))]
use crate::{
    CF_AI_ATTACHMENTS, CF_AI_CHATS, CF_AI_CREDITS, CF_AI_TRANSACTIONS, CF_AI_TREASURY, CF_BALANCES, CF_BANNED_PEERS,
    CF_BLOCK_HASH_TO_HEIGHT, CF_BLOCKS, CF_BULLSHARK_CERT, CF_DAG_VERTICES, CF_MANIFEST,
    CF_NARWHAL_PAYLOADS, CF_PAYMENT_LOCKS, CF_PAYMENT_PROPOSALS, CF_PAYMENT_VOTES, CF_TRANSACTIONS,
};

/// Async KV store trait for storage abstraction
#[async_trait]
pub trait KVStore: Send + Sync {
    /// Put key-value pair in column family
    async fn put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()>;

    /// Put key-value pair in column family with SYNC (fsync to disk - survives hard kills)
    async fn put_sync(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()>;

    /// Get value by key from column family
    async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Delete key from column family
    async fn delete(&self, cf: &str, key: &[u8]) -> Result<()>;

    /// Write atomic batch across column families
    async fn write_batch(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()>;

    /// Write atomic batch in BULK MODE (no fsync, optimized for initial sync)
    /// WARNING: Data loss risk on crash - only use during initial blockchain sync
    async fn write_batch_bulk(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()>;

    /// 🚀 v1.0.89-beta: TURBO MODE - Write batch with WAL but NO per-write fsync
    /// Best of both worlds:
    /// - WAL enabled (crash recovery possible)
    /// - NO fsync per write (~0.1ms vs 2-5ms)
    /// - Caller must call sync_wal() periodically (every 1-2 seconds)
    /// - On crash: lose up to 1-2 seconds of blocks (re-fetchable from peers)
    /// Target: 1000+ blocks/second during initial sync
    async fn write_batch_turbo(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()>;

    /// Scan keys with prefix in column family
    async fn scan_prefix(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Scan all keys in column family (use with caution)
    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Flush writes to disk
    async fn flush(&self) -> Result<()>;

    /// Compact database
    async fn compact(&self) -> Result<()>;

    /// Get database size in bytes
    async fn get_db_size(&self) -> Result<u64>;

    /// 🚨 v0.9.60-beta: CRITICAL DURABILITY ADDITIONS

    /// Create checkpoint (hard-linked snapshot) for instant, consistent backups
    /// Uses RocksDB Checkpoint API - zero-copy, crash-safe
    async fn create_checkpoint(&self, checkpoint_dir: &str) -> Result<()>;

    /// Sync WAL to disk (call before shutdown for maximum safety)
    /// Forces all pending writes to be durably persisted
    async fn sync_wal(&self) -> Result<()>;

    /// Graceful shutdown with full data persistence
    /// Syncs WAL + flushes all memtables + closes DB safely
    async fn shutdown_gracefully(&self) -> Result<()>;

    /// Verify backup integrity (read checksum validation)
    /// Call after creating checkpoint to ensure it's not corrupted
    async fn verify_checkpoint(&self, checkpoint_dir: &str) -> Result<bool>;

    /// 🚀 v1.0.100-beta: Multi-get for batch fetching (10-50x faster than N individual gets)
    /// Returns a Vec of Option<Vec<u8>> in the same order as keys
    async fn multi_get(&self, cf: &str, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>>;
}

/// RocksDB implementation optimized for DagKnight workloads (Linux/macOS only)
#[cfg(not(target_os = "windows"))]
pub struct RocksDBKV {
    db: Arc<DB>,
    /// Store DB path for column family lookup instead of raw CF handles
    db_path: String,
    /// Quantum RNG for encryption keys (Phase 2+)
    qrng: Option<Arc<QuantumRNG>>,
    /// Current cryptographic phase
    phase: Phase,
    /// Adaptive pruning configuration
    pub pruning_config: crate::pruning::PruningConfig,
    /// 🔐 v1.0.43-beta: RocksDB encryption-at-rest with ZK-STARK untrusted setup
    encryption_manager: Option<Arc<crate::encryption::EncryptionManager>>,
}

#[cfg(not(target_os = "windows"))]
impl RocksDBKV {
    /// Open hot database with optimized settings for frequent access
    pub async fn open_hot_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_hot_db_with_phase(path, Phase::Phase0).await
    }

    /// Open hot database with specific phase support
    pub async fn open_hot_db_with_phase<P: AsRef<Path>>(path: P, phase: Phase) -> Result<Self> {
        let path = path.as_ref();
        info!("🔥 Opening hot RocksDB at {:?} for {:?}", path, phase);

        // 🔐 v1.0.43-beta: Initialize encryption if environment variables are set
        let encryption_manager = Self::initialize_encryption_if_enabled()?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Hot DB optimizations - reduced threading to fix glibc TLS allocation issue
        let bg_jobs = std::env::var("ROCKSDB_MAX_BACKGROUND_JOBS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let bg_compactions = std::env::var("ROCKSDB_MAX_COMPACTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let bg_flushes = std::env::var("ROCKSDB_MAX_FLUSHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        info!(
            "🗄️ RocksDB Hot DB Threading: jobs={}, compactions={}, flushes={}",
            bg_jobs, bg_compactions, bg_flushes
        );

        opts.set_max_background_jobs(bg_jobs); // Reduced from 8 to avoid TLS allocation failures
        opts.set_max_background_compactions(bg_compactions); // Limit compaction threads
        opts.set_max_background_flushes(bg_flushes); // Limit flush threads

        // 🚀 OPTIMIZED FOR BULK SYNC PERFORMANCE
        // Check if TURBO_SYNC environment variable is set for bulk import mode
        let turbo_sync_mode = std::env::var("TURBO_SYNC_ENABLED").is_ok();

        if turbo_sync_mode {
            info!("🚀 TURBO SYNC MODE ENABLED - Optimizing RocksDB for bulk writes");
            // 🚀 v1.4.2-beta: Enhanced sync optimization from SYNC_OPTIMIZATION_TECHNICAL_REVIEW.md
            opts.set_write_buffer_size(512 * 1024 * 1024); // 512MB for bulk imports (doubled)
            opts.set_max_write_buffer_number(8); // More buffers to avoid stalls
            opts.set_target_file_size_base(256 * 1024 * 1024); // 256MB SST files
            opts.set_level_zero_file_num_compaction_trigger(16); // Delay compaction during sync
            opts.set_level_zero_slowdown_writes_trigger(32);
            opts.set_level_zero_stop_writes_trigger(64);

            // 🚀 v1.4.2-beta: Add LRU block cache (2GB) for faster reads during sync
            // This reduces read amplification when verifying blocks
            let cache_size = std::env::var("ROCKSDB_BLOCK_CACHE_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2048) * 1024 * 1024;
            let block_cache = rocksdb::Cache::new_lru_cache(cache_size);
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_block_cache(&block_cache);
            block_opts.set_bloom_filter(10.0, true); // 10-bit bloom filter for faster lookups
            opts.set_block_based_table_factory(&block_opts);

            info!("🗄️ RocksDB block cache: {} MB", cache_size / (1024 * 1024));
        } else {
            // 🔧 v1.0.21-beta: OPTIMIZED FOR CONTINUOUS BLOCK PRODUCTION
            // Problem: Progressive degradation (170ms → 994ms over time)
            // Root cause: Compaction falling behind write rate
            // Fix: Larger buffers + more aggressive compaction
            opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB (doubled from 64MB)
            opts.set_max_write_buffer_number(6); // 6 buffers (increased from 4)
            opts.set_target_file_size_base(128 * 1024 * 1024); // 128MB SST files (doubled)
            opts.set_level_zero_file_num_compaction_trigger(2); // More aggressive (was 4)
            opts.set_level_zero_slowdown_writes_trigger(6); // Earlier slowdown warning (was 8)
            opts.set_level_zero_stop_writes_trigger(10); // Earlier stop (was 16)
        }

        // 🚨 v0.9.60-beta: MAXIMUM DURABILITY MODE (5 phases of corruption → NEVER AGAIN!)
        // ChatGPT-recommended hardened RocksDB settings for mainnet-grade reliability

        // ========== DURABILITY SETTINGS (CRASH-SAFE) ==========
        opts.set_use_fsync(true); // use fsync() not fdatasync() - strongest guarantee
        opts.set_paranoid_checks(true); // Detect corruption early, fail loud
        opts.set_atomic_flush(true); // Multi-CF consistency (all or nothing)
        // 🚨 v1.0.79-beta: CRITICAL FIX - Change from PointInTime to TolerateCorruptedTailRecords
        //
        // ROOT CAUSE OF DATA LOSS:
        // - PointInTime mode stops at the FIRST inconsistency in the WAL
        // - On kill -9, the last WAL record is often partial/corrupted
        // - PointInTime truncates EVERYTHING after that point, losing many valid blocks
        //
        // FIX: TolerateCorruptedTailRecords
        // - Only discards corrupted records at the very END of the WAL
        // - Recovers all valid records before the corruption
        // - Ideal for kill -9 / power failure scenarios
        //
        // WAL Recovery Modes (from safest to most aggressive):
        // 1. AbsoluteConsistency - Fails on ANY WAL corruption (too strict)
        // 2. PointInTime - Stops at first inconsistency (CAUSES DATA LOSS - old setting)
        // 3. TolerateCorruptedTailRecords - Tolerates corrupted tail (BEST FOR BLOCKCHAIN)
        // 4. SkipAnyCorruptedRecords - Skips any corrupted records (too loose)
        //
        opts.set_wal_recovery_mode(rocksdb::DBRecoveryMode::TolerateCorruptedTailRecords);

        // ========== WAL (Write-Ahead Log) PROTECTION ==========
        opts.set_wal_ttl_seconds(300); // 5 minutes - delete after flush
        opts.set_wal_size_limit_mb(256); // 256MB max - prevents unbounded growth
        opts.set_max_total_wal_size(64 * 1024 * 1024); // 64MB total WAL budget
        // ChatGPT P0: DO NOT set manual_wal_flush(true) - auto flush is safer
        // opts.set_manual_wal_flush(true); // DISABLED per ChatGPT recommendation

        // ========== STEADY IO (PREVENT BURST CORRUPTION) ==========
        opts.set_bytes_per_sync(1024 * 1024); // 1 MiB - sync data in steady chunks
        opts.set_wal_bytes_per_sync(1024 * 1024); // 1 MiB - sync WAL in steady chunks

        // ========== MEMORY BUDGET (FORCE FLUSHES) ==========
        // 🔧 v1.0.21-beta: Increased from 128MB to match new write buffer settings
        // With 6 buffers × 128MB each = 768MB potential, limit total to 384MB
        opts.set_db_write_buffer_size(384 * 1024 * 1024); // 384MB total memtable budget (tripled)

        // 🚨 THE SILVER BULLET: Force flushes on shutdown (RocksDB 7+ defaults to skip!)
        // This was the root cause - graceful shutdowns avoided flushes, relied on WAL
        // When WAL exceeded limits or got corrupted → 100% data loss
        // NOTE: set_avoid_flush_during_shutdown() not available in rust-rocksdb 0.22.0
        // WORKAROUND: Manual flush_cf() calls + smaller write buffers + WAL limits
        // opts.set_avoid_flush_during_shutdown(false); // Would be ideal if available

        // Initialize quantum encryption for Phase 2+
        let qrng = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🌌 Initializing quantum RNG for storage encryption");
            let config = QRNGConfig {
                min_entropy_quality: 0.99, // Highest quality for encryption
                pool_size: 8192,
                polling_interval_ms: 100,
                ..Default::default()
            };

            match QuantumRNG::new(phase, config).await {
                Ok(qrng) => {
                    info!("✅ Quantum RNG initialized for storage encryption");
                    Some(Arc::new(qrng))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize storage QRNG: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let cfs = vec![
            Self::create_blocks_cf(),
            Self::create_dag_vertices_cf(),
            Self::create_bullshark_cert_cf(),
            Self::create_manifest_cf(),
            Self::create_transactions_cf(),
            Self::create_balances_cf(),  // v0.8.2-beta: Balance consensus storage
            Self::create_block_hash_to_height_cf(),  // v0.8.3-beta: Block hash index
            Self::create_ai_chats_cf(),
            Self::create_ai_credits_cf(),
            Self::create_ai_transactions_cf(),
            Self::create_ai_treasury_cf(),
            Self::create_ai_attachments_cf(),  // v0.9.9-beta: AI chat attachments
            Self::create_payment_proposals_cf(),
            Self::create_payment_votes_cf(),
            Self::create_payment_locks_cf(),
            Self::create_banned_peers_cf(),  // v0.9.7-beta: ZK proof ban persistence
            Self::create_sync_certificates_cf(),  // v0.9.18-beta: TurboSync AEGIS-QL certificates
            Self::create_peer_trust_cf(),  // v0.9.18-beta: AEGIS-QL peer trust metrics
            Self::create_processed_updates_cf(),  // ✅ v0.9.98-beta: P2P durability idempotency tracking
            // ========== v1.0.60-beta: Comprehensive State Sync CFs ==========
            Self::create_state_trie_cf(),        // Sparse Merkle trie for state roots
            Self::create_token_balances_cf(),    // Token balances (all tokens including QUG/QUGUSD)
            Self::create_tokens_cf(),            // Token metadata
            Self::create_dex_pools_cf(),         // DEX liquidity pools
            Self::create_lp_balances_cf(),       // LP token balances
            Self::create_vaults_cf(),            // Collateral vaults
            Self::create_oracle_prices_cf(),     // Oracle price feeds
            Self::create_contracts_cf(),         // Smart contracts
            Self::create_contract_storage_cf(),  // Contract storage
            Self::create_ai_credits_v2_cf(),     // AI credits v2 (with stats)
            Self::create_stakes_cf(),            // Staking positions
            Self::create_nonces_cf(),            // Account nonces
            // ========== v1.4.2-beta: QNO Prediction Staking ==========
            Self::create_qno_stakes_cf(),        // QNO staking positions
            Self::create_qno_domains_cf(),       // QNO prediction domains
            Self::create_qno_stats_cf(),         // QNO global statistics
            // ========== v2.3.9-beta: Swap History ==========
            Self::create_swap_history_cf(),      // Swap/transaction history for Token Details Modal
            // ========== v2.7.9-beta: Perpetual Trading ==========
            Self::create_perp_positions_cf(),    // Perpetual futures positions
            Self::create_perp_orders_cf(),       // Perpetual futures orders
            Self::create_perp_trades_cf(),       // Perpetual futures trade history
            Self::create_perp_funding_cf(),      // Perpetual funding rate history
            Self::create_perp_liquidations_cf(), // Perpetual liquidation records
            // ========== v3.5.8-beta: Wallet Transaction History ==========
            Self::create_wallet_tx_index_cf(),   // Wallet-indexed transaction history
            Self::create_wallet_swap_index_cf(), // Wallet-indexed DEX swap history
            // ========== v3.6.0-beta: Price History ==========
            Self::create_price_history_cf(),     // Consensus-verified price history
            // ========== v3.9.1-beta: Bank Messaging & Identity ==========
            Self::create_bank_messages_cf(),     // Bank messages storage
            Self::create_bank_msg_index_cf(),    // Bank message index by wallet
            Self::create_user_identities_cf(),   // User identity records
            Self::create_death_certificates_cf(), // Death certificates for inheritance
        ];

        let mut kv = Self::open_with_cfs(path, opts, cfs).await?;
        kv.qrng = qrng;
        kv.phase = phase;
        kv.encryption_manager = encryption_manager;

        Ok(kv)
    }

    /// Open cold database with optimized settings for large payloads
    pub async fn open_cold_db<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("🧊 Opening cold RocksDB at {:?}", path);

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Cold DB optimizations - reduced threading to fix glibc TLS allocation issue
        let cold_bg_jobs = std::env::var("ROCKSDB_COLD_MAX_BACKGROUND_JOBS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let cold_bg_compactions = std::env::var("ROCKSDB_COLD_MAX_COMPACTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let cold_bg_flushes = std::env::var("ROCKSDB_COLD_MAX_FLUSHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        info!(
            "🧊 RocksDB Cold DB Threading: jobs={}, compactions={}, flushes={}",
            cold_bg_jobs, cold_bg_compactions, cold_bg_flushes
        );

        opts.set_max_background_jobs(cold_bg_jobs); // Reduced from 4 to avoid TLS allocation failures
        opts.set_max_background_compactions(cold_bg_compactions); // Single compaction thread
        opts.set_max_background_flushes(cold_bg_flushes); // Single flush thread
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
        opts.set_max_write_buffer_number(2);
        opts.set_target_file_size_base(256 * 1024 * 1024); // 256MB
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // 🚨 v0.9.60-beta: COLD DB DURABILITY (same as hot DB)
        opts.set_use_fsync(true);
        opts.set_paranoid_checks(true);
        opts.set_atomic_flush(true);
        opts.set_bytes_per_sync(1024 * 1024); // 1 MiB
        opts.set_wal_bytes_per_sync(1024 * 1024); // 1 MiB

        let cfs = vec![Self::create_narwhal_payloads_cf()];

        Self::open_with_cfs(path, opts, cfs).await
    }

    /// Open RocksDB with column families
    async fn open_with_cfs<P: AsRef<Path>>(
        path: P,
        opts: Options,
        cfs: Vec<ColumnFamilyDescriptor>,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // ✅ v0.9.34-beta: Automatic column family migration for existing databases
        // First, try to list existing column families to detect if migration is needed
        let existing_cfs = DB::list_cf(&Options::default(), &path)
            .unwrap_or_else(|_| vec!["default".to_string()]); // New DB case

        // Check which requested CFs are missing
        let requested_cf_names: Vec<String> = cfs.iter()
            .map(|cf| cf.name().to_string())
            .collect();

        let missing_cfs: Vec<String> = requested_cf_names.iter()
            .filter(|name| !existing_cfs.contains(name))
            .cloned()
            .collect();

        // v2.3.9-beta: Debug logging for migration detection
        info!("🔍 [CF-CHECK] Existing CFs: {}, Requested CFs: {}, Missing: {}",
            existing_cfs.len(), requested_cf_names.len(), missing_cfs.len());
        if !missing_cfs.is_empty() {
            info!("📋 [CF-CHECK] Missing column families: {:?}", missing_cfs);
        }

        if !missing_cfs.is_empty() && existing_cfs.len() > 1 {
            // Database exists but is missing some column families - perform migration
            warn!("⚠️  [AUTO-MIGRATION] Database exists but missing {} column families", missing_cfs.len());
            info!("📋 [AUTO-MIGRATION] Missing CFs: {:?}", missing_cfs);
            info!("🔧 [AUTO-MIGRATION] Performing automatic migration...");

            // Open DB with only existing column families
            let existing_cf_descriptors: Vec<ColumnFamilyDescriptor> = existing_cfs.iter()
                .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
                .collect();

            let db = DB::open_cf_descriptors(&opts, &path, existing_cf_descriptors)
                .context("Failed to open RocksDB for migration")?;

            // Create missing column families with default options
            for cf_name in &missing_cfs {
                info!("➕ [AUTO-MIGRATION] Creating column family: {}", cf_name);

                let mut cf_opts = Options::default();
                cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                cf_opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
                cf_opts.set_max_write_buffer_number(2);

                db.create_cf(cf_name, &cf_opts)
                    .context(format!("Failed to create column family: {}", cf_name))?;
                info!("✅ [AUTO-MIGRATION] Column family '{}' created successfully", cf_name);
            }

            info!("🎉 [AUTO-MIGRATION] Column family migration complete!");
            info!("   Your node has been automatically upgraded");

            Ok(Self {
                db: Arc::new(db),
                db_path: path_str,
                qrng: None,           // Will be set by caller
                phase: Phase::Phase0, // Will be set by caller
                pruning_config: crate::pruning::PruningConfig::default(),
                encryption_manager: None, // Will be set by caller
            })
        } else {
            // Normal path - database is new or already has all CFs
            let db = DB::open_cf_descriptors(&opts, &path, cfs).context("Failed to open RocksDB")?;

            Ok(Self {
                db: Arc::new(db),
                db_path: path_str,
                qrng: None,           // Will be set by caller
                phase: Phase::Phase0, // Will be set by caller
                pruning_config: crate::pruning::PruningConfig::default(),
                encryption_manager: None, // Will be set by caller
            })
        }
    }

    /// Get the underlying RocksDB handle for direct access
    /// Used by components that need raw RocksDB access (e.g., TokenRegistry, PriceHistoryManager)
    pub fn get_raw_db(&self) -> Arc<DB> {
        self.db.clone()
    }

    /// Create blocks column family (height || hash -> block)
    fn create_blocks_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_block_based_table_factory(&rocksdb::BlockBasedOptions::default());

        // 🚨 CRITICAL FIX v0.7.3-beta: Force persistent flushes (Expert-reviewed)
        // Root cause: avoid_flush_during_shutdown=true + unlimited WAL = data loss
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB - balanced (~1600 blocks/flush)
        opts.set_min_write_buffer_number_to_merge(1); // Flush immediately
        opts.set_max_write_buffer_number(3); // Triple buffering
        opts.set_disable_auto_compactions(false); // Enable auto compactions
        opts.set_level_zero_file_num_compaction_trigger(2); // Compact aggressively

        // 🚨 SILVER BULLET: Force flushes on shutdown (RocksDB 7.x+ defaults to true!)
        // This was THE bug - graceful shutdowns skipped flushes, relied on WAL recovery
        // Note: avoid_flush_during_shutdown not available in rust-rocksdb 0.22.0
        // Workaround: Smaller buffers + bounded WAL + explicit flushes

        // 🚨 Additional safety settings
        opts.set_paranoid_checks(true); // Extra validation

        // Background tuning
        opts.set_max_background_flushes(2);
        opts.set_max_background_compactions(2);

        ColumnFamilyDescriptor::new(CF_BLOCKS, opts)
    }

    /// Create DAG vertices column family (round || author || seq -> vertex)
    fn create_dag_vertices_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Enable prefix seek for round-based queries
        // Use fixed prefix length of 8 bytes for round number
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8));

        let mut table_opts = rocksdb::BlockBasedOptions::default();
        table_opts.set_index_type(rocksdb::BlockBasedIndexType::HashSearch);
        opts.set_block_based_table_factory(&table_opts);

        ColumnFamilyDescriptor::new(CF_DAG_VERTICES, opts)
    }

    /// Create Bullshark certificates column family (round -> certificate)
    fn create_bullshark_cert_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Snappy);

        ColumnFamilyDescriptor::new(CF_BULLSHARK_CERT, opts)
    }

    /// Create manifest column family (metadata -> value)
    fn create_manifest_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::None); // Small data

        ColumnFamilyDescriptor::new(CF_MANIFEST, opts)
    }

    /// Create transactions column family (tx_id -> transaction)
    fn create_transactions_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB write buffer
        opts.set_target_file_size_base(128 * 1024 * 1024); // 128MB target file size

        ColumnFamilyDescriptor::new(CF_TRANSACTIONS, opts)
    }

    /// Create balances column family (wallet_address -> balance)
    /// v0.8.1-beta: Balance consensus storage for mining rewards and transfers
    fn create_balances_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB write buffer (frequent updates)
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB target file size

        // Optimize for frequent balance updates
        opts.set_max_write_buffer_number(3); // Triple buffering for high write load
        opts.set_level_zero_file_num_compaction_trigger(4); // Compact when 4 files accumulate

        ColumnFamilyDescriptor::new(CF_BALANCES, opts)
    }

    /// Create block hash to height column family (block_hash -> height)
    /// v0.8.3-beta: Block hash index for efficient block lookups by hash
    fn create_block_hash_to_height_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB write buffer
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB target file size

        // Optimize for read-heavy workload (hash lookups)
        opts.set_max_write_buffer_number(2); // Dual buffering sufficient
        opts.set_level_zero_file_num_compaction_trigger(4);

        ColumnFamilyDescriptor::new(CF_BLOCK_HASH_TO_HEIGHT, opts)
    }

    /// Create AI chats column family (chat:* keys -> chat data)
    fn create_ai_chats_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression for text

        // Enable prefix seek for chat-based queries
        // Use fixed prefix length of 5 bytes for "chat:" prefix
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5));

        let mut table_opts = rocksdb::BlockBasedOptions::default();
        table_opts.set_index_type(rocksdb::BlockBasedIndexType::HashSearch);
        opts.set_block_based_table_factory(&table_opts);

        ColumnFamilyDescriptor::new(CF_AI_CHATS, opts)
    }

    /// Create AI credits column family (credits:* -> wallet credits)
    fn create_ai_credits_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8)); // "credits:"

        ColumnFamilyDescriptor::new(CF_AI_CREDITS, opts)
    }

    /// Create AI transactions column family (aitx:* -> transaction records)
    fn create_ai_transactions_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5)); // "aitx:"

        ColumnFamilyDescriptor::new(CF_AI_TRANSACTIONS, opts)
    }

    /// Create AI treasury column family (treasury:master -> master wallet balance)
    fn create_ai_treasury_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Single key "treasury:master" - no prefix needed

        ColumnFamilyDescriptor::new(CF_AI_TREASURY, opts)
    }

    /// Create AI attachments column family (attachment:* -> metadata) - v0.9.9-beta
    fn create_ai_attachments_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(11)); // "attachment:"

        ColumnFamilyDescriptor::new(CF_AI_ATTACHMENTS, opts)
    }

    /// Create payment proposals column family (proposal:* -> payment proposals)
    fn create_payment_proposals_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(9)); // "proposal:"

        ColumnFamilyDescriptor::new(CF_PAYMENT_PROPOSALS, opts)
    }

    /// Create payment votes column family (vote:* -> validator votes)
    fn create_payment_votes_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5)); // "vote:"

        ColumnFamilyDescriptor::new(CF_PAYMENT_VOTES, opts)
    }

    /// Create payment locks column family (lock:* -> payment locks)
    fn create_payment_locks_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5)); // "lock:"

        ColumnFamilyDescriptor::new(CF_PAYMENT_LOCKS, opts)
    }

    /// Create banned peers column family (peer_id -> BanRecord) - v0.9.7-beta
    /// Stores persistent ban list for ZK proof verification failures
    fn create_banned_peers_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (PeerId + timestamp + reason), optimize for reads
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB
        opts.set_max_write_buffer_number(2);

        ColumnFamilyDescriptor::new(CF_BANNED_PEERS, opts)
    }

    /// Create Narwhal payloads column family (digest -> payload)
    fn create_narwhal_payloads_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Large values, optimize for sequential writes
        opts.set_write_buffer_size(256 * 1024 * 1024); // 256MB
        opts.set_max_write_buffer_number(2);
        opts.set_target_file_size_base(512 * 1024 * 1024); // 512MB

        ColumnFamilyDescriptor::new(CF_NARWHAL_PAYLOADS, opts)
    }

    /// Create sync certificates column family (sync_id -> SyncCertificate) - v0.9.18-beta
    /// Stores AEGIS-QL sync affirmation certificates for TurboSync
    fn create_sync_certificates_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small-medium values (certificates), optimize for reads during sync
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);

        ColumnFamilyDescriptor::new(crate::CF_SYNC_CERTIFICATES, opts)
    }

    /// Create peer trust column family (peer_id -> TrustMetrics) - v0.9.18-beta
    /// Stores AEGIS-QL peer trust metrics for sync reliability
    fn create_peer_trust_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (metrics), optimize for frequent updates
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB
        opts.set_max_write_buffer_number(2);

        ColumnFamilyDescriptor::new(crate::CF_PEER_TRUST, opts)
    }

    /// Create processed updates column family (update_id -> timestamp) - v0.9.98-beta
    /// Stores processed update IDs for P2P durability and idempotency
    /// AI Expert Consensus: Required to prevent duplicate processing of gossipsub messages
    fn create_processed_updates_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (timestamps), optimize for fast lookups
        opts.set_write_buffer_size(4 * 1024 * 1024); // 4MB
        opts.set_max_write_buffer_number(2);
        // TTL could be added in future for automatic cleanup of old update IDs

        ColumnFamilyDescriptor::new("processed_updates", opts)
    }

    // ========== v1.0.60-beta: Comprehensive State Sync Column Families ==========

    /// State Merkle Trie nodes for cryptographic state root verification
    fn create_state_trie_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Trie nodes are 65 bytes (internal) or 65 bytes (leaf)
        // Optimize for random reads (proof generation)
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(4);
        opts.optimize_for_point_lookup(128 * 1024 * 1024); // 128MB bloom filter cache
        ColumnFamilyDescriptor::new(crate::sparse_merkle_trie::CF_STATE_TRIE, opts)
    }

    /// Token balances (account + token -> balance)
    fn create_token_balances_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Keys are 64 bytes (account + token), values are 8 bytes (u64 balance)
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(4);
        opts.optimize_for_point_lookup(256 * 1024 * 1024); // 256MB bloom filter
        ColumnFamilyDescriptor::new(crate::CF_TOKEN_BALANCES, opts)
    }

    /// Token metadata (token address -> metadata)
    fn create_tokens_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_TOKENS, opts)
    }

    /// DEX liquidity pools (pool_id -> pool state)
    fn create_dex_pools_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_DEX_POOLS, opts)
    }

    /// LP token balances (pool_id + account -> LP balance)
    fn create_lp_balances_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_LP_BALANCES, opts)
    }

    /// Collateral vaults (vault_id -> vault state)
    fn create_vaults_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_VAULTS, opts)
    }

    /// Oracle price feeds (feed_id -> price data)
    fn create_oracle_prices_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_ORACLE_PRICES, opts)
    }

    /// Smart contracts (contract_address -> code hash + metadata)
    fn create_contracts_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_CONTRACTS, opts)
    }

    /// Contract storage (contract_address + key -> value)
    fn create_contract_storage_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Largest CF - contract storage can be huge
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
        opts.set_max_write_buffer_number(4);
        ColumnFamilyDescriptor::new(crate::CF_CONTRACT_STORAGE, opts)
    }

    /// AI credits (account -> credits balance + stats)
    fn create_ai_credits_v2_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_AI_CREDITS_V2, opts)
    }

    /// Staking positions (staker + validator -> stake info)
    fn create_stakes_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_STAKES, opts)
    }

    /// Account nonces for replay protection (account -> nonce)
    fn create_nonces_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        opts.optimize_for_point_lookup(64 * 1024 * 1024); // Fast nonce lookups
        ColumnFamilyDescriptor::new(crate::CF_NONCES, opts)
    }

    // ========== QNO (Quantum Neural Oracle) Column Families ==========

    /// QNO staking positions: wallet_address:stake_id -> StakingPosition JSON
    fn create_qno_stakes_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(3);
        ColumnFamilyDescriptor::new(crate::CF_QNO_STAKES, opts)
    }

    /// QNO prediction domains: domain_id -> PredictionDomain JSON
    fn create_qno_domains_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB - small, rarely changes
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_QNO_DOMAINS, opts)
    }

    /// QNO global statistics: "global" -> StakingStats JSON
    fn create_qno_stats_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(4 * 1024 * 1024); // 4MB - single key
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_QNO_STATS, opts)
    }

    // ========== v2.3.9-beta: Swap History Column Family ==========

    /// Swap history for Token Details Modal transaction history
    /// Key format: "swap:{token}:{timestamp}:{tx_id}" -> JSON swap record
    fn create_swap_history_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB - many small records
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8)); // "swap:{T}"
        ColumnFamilyDescriptor::new(crate::CF_SWAP_HISTORY, opts)
    }

    // ========== v2.7.9-beta: Perpetual Trading Column Families ==========

    fn create_perp_positions_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_PERP_POSITIONS, opts)
    }

    fn create_perp_orders_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_PERP_ORDERS, opts)
    }

    fn create_perp_trades_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_PERP_TRADES, opts)
    }

    fn create_perp_funding_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_PERP_FUNDING, opts)
    }

    fn create_perp_liquidations_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_PERP_LIQUIDATIONS, opts)
    }

    /// v3.5.8-beta: Wallet transaction index for decentralized history
    fn create_wallet_tx_index_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(3);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        ColumnFamilyDescriptor::new(crate::CF_WALLET_TX_INDEX, opts)
    }

    /// v3.5.8-beta: Wallet swap index for decentralized DEX history
    fn create_wallet_swap_index_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        ColumnFamilyDescriptor::new(crate::CF_WALLET_SWAP_INDEX, opts)
    }

    /// v3.6.0-beta: Consensus-verified price history for tokens
    /// Key format: [token_address:32][inverted_timestamp:8] for reverse chronological order
    fn create_price_history_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB - many small price snapshots
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are token address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        ColumnFamilyDescriptor::new(crate::CF_PRICE_HISTORY, opts)
    }

    /// v3.9.1-beta: Bank messages storage
    /// Key format: msg_id (string), Value: BankMessage JSON
    fn create_bank_messages_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_BANK_MESSAGES, opts)
    }

    /// v3.9.1-beta: Bank message index by wallet
    /// Key format: [wallet:32][inverted_timestamp:8], Value: msg_id
    fn create_bank_msg_index_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        ColumnFamilyDescriptor::new(crate::CF_BANK_MSG_INDEX, opts)
    }

    /// v3.9.1-beta: User identity records
    /// Key format: wallet_address (string), Value: UserIdentity JSON
    fn create_user_identities_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_USER_IDENTITIES, opts)
    }

    /// v3.9.1-beta: Death certificates for inheritance
    /// Key format: cert_id (string), Value: DeathCertificate JSON
    fn create_death_certificates_cf() -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        ColumnFamilyDescriptor::new(crate::CF_DEATH_CERTIFICATES, opts)
    }

    /// Get column family handle (public for transactions - v0.8.1-beta)
    pub fn get_cf(&self, cf_name: &str) -> Result<Arc<rocksdb::BoundColumnFamily>> {
        self.db
            .cf_handle(cf_name)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", cf_name))
    }
}

#[cfg(not(target_os = "windows"))]
#[async_trait]
impl KVStore for RocksDBKV {
    async fn put(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let cf_handle = self.get_cf(cf)?;

        // v0.9.93-beta P0 FIX: ALWAYS use sync=true for durability
        // Kimi AI was correct - unsync'd puts caused "blocks saved but missing" corruption
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.set_sync(true); // Force fsync() to survive kill -9
        write_opts.disable_wal(false); // Keep WAL enabled

        self.db
            .put_cf_opt(&cf_handle, key, value, &write_opts)
            .context("RocksDB put failed")?;

        debug!("💾 Synced put: cf={}, key_len={}", cf, key.len());

        Ok(())
    }

    async fn put_sync(&self, cf: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let cf_handle = self.get_cf(cf)?;

        // Create write options with sync=true to force fsync to disk
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.set_sync(true); // CRITICAL: Force fsync() syscall - survives hard kills
        write_opts.disable_wal(false); // Keep WAL enabled

        self.db
            .put_cf_opt(&cf_handle, key, value, &write_opts)
            .context("RocksDB synced put failed")?;

        // REMOVED flush_cf() - immediate flush deletes WAL prematurely!
        // WAL with fsync is sufficient for durability. RocksDB will flush memtable
        // to SST naturally, and WAL will be preserved until flush completes.

        Ok(())
    }

    async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf_handle = self.get_cf(cf)?;

        let result = self
            .db
            .get_cf(&cf_handle, key)
            .context("RocksDB get failed")?;

        Ok(result)
    }

    async fn delete(&self, cf: &str, key: &[u8]) -> Result<()> {
        let cf_handle = self.get_cf(cf)?;

        // v0.9.93-beta P0 FIX: ALWAYS use sync=true for durability
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.set_sync(true); // Force fsync() to survive kill -9
        write_opts.disable_wal(false); // Keep WAL enabled

        self.db
            .delete_cf_opt(&cf_handle, key, &write_opts)
            .context("RocksDB delete failed")?;

        debug!("🗑️  Synced delete: cf={}, key_len={}", cf, key.len());

        Ok(())
    }

    async fn write_batch(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();

        // PHASE 1: Prepare WriteBatch in async context (cheap, no blocking)
        let mut write_batch = WriteBatch::default();
        let mut cf_names_to_flush: Vec<&str> = Vec::new();

        for (cf_name, key, value) in &batch {
            let cf_handle = self.get_cf(cf_name)?;
            write_batch.put_cf(&cf_handle, key, value);
            if !cf_names_to_flush.contains(cf_name) {
                cf_names_to_flush.push(cf_name);
            }
        }

        // Clone Arc for move into blocking context
        let db = self.db.clone();
        // Convert &str to String for move into closure
        let cf_names_owned: Vec<String> = cf_names_to_flush.into_iter().map(|s| s.to_string()).collect();

        // 🚨 CRITICAL FIX v0.9.94-beta: DEADLOCK RESOLUTION
        // Problem: write_opt() with sync=true and flush_cf_opt() with wait=true
        //          are BLOCKING operations that stall Tokio executor threads.
        //          This caused BlockWriter to stop receiving messages after ~23 minutes.
        // Solution: Move ALL RocksDB blocking operations to spawn_blocking.
        // Expert consensus: ChatGPT, Kimi AI, DeepSeek all agree (95% confidence)
        // Why this works:
        //   - spawn_blocking runs on dedicated blocking thread pool
        //   - Tokio executor threads stay free to poll async tasks
        //   - BlockWriter can continue receiving channel messages
        //   - No change to durability guarantees (sync=true preserved)
        // References:
        //   - https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html
        //   - https://stackoverflow.com/q/66087127 (Tokio blocking I/O guidance)

        tokio::task::spawn_blocking(move || {
            let blocking_start = Instant::now();

            // CRITICAL FIX: Use synced write options to prevent data loss on hard kills
            let mut write_opts = rocksdb::WriteOptions::default();
            write_opts.set_sync(true); // Force fsync() to survive hard kills (pkill -9, service restart)
            write_opts.disable_wal(false); // Keep WAL enabled for crash recovery

            // BLOCKING OPERATION #1: Write batch with fsync
            db.write_opt(write_batch, &write_opts)
                .context("RocksDB batch write failed")?;

            debug!("✅ RocksDB write_opt completed in {:?}", blocking_start.elapsed());

            // 🚨 v0.9.97-beta: CRITICAL FIX - Remove flush_cf() from hot path
            // ChatGPT Expert Analysis (95% confidence):
            // "Remove flush_cf() from the hot path. Flushing moves memtables to SSTs but
            //  does not add crash durability beyond the WAL; it can even add latency/jitter.
            //  If you want a belt-and-suspenders, call db.sync_wal() (redundant if sync=true
            //  was used, but harmless), not flush_cf()."
            //
            // Performance Impact:
            // - Before: 3-5ms per write (with flush_cf)
            // - After: 1-2ms per write (without flush_cf)
            // - Durability: SAME (WAL with fsync is the durability barrier)
            //
            // Why flush_cf() was removed:
            // 1. write_opt() with set_sync(true) already guarantees fsync() to WAL
            // 2. WAL is the crash recovery mechanism, not SST files
            // 3. flush_cf() is an I/O-heavy compaction operation (memtable → SST)
            // 4. It adds 1-3ms latency with zero durability benefit
            // 5. Background compaction will flush memtables automatically
            //
            // The sync=true guarantee from WriteOptions is sufficient:
            // - WAL is fsync'd to disk before write_opt() returns
            // - On crash/restart, RocksDB replays WAL to recover memtable
            // - SST files are just an optimization, not durability mechanism

            // 🚨 v1.0.78-beta: BATCHED FLUSH for crash safety + performance
            // LESSON LEARNED: WAL + set_sync(true) is NOT sufficient for kill -9 durability!
            //
            // Root cause: PointInTime WAL recovery can truncate WAL, causing data loss
            //
            // SOLUTION: Batched flush every N writes
            // - Flush every 100 blocks during sync (max 100 blocks lost on kill -9)
            // - Fast sync: ~500 blocks/sec (vs ~20 blocks/sec with per-block flush)
            // - Acceptable trade-off: lose max 100 blocks vs lose 600+ blocks
            //
            // The counter is stored in a static atomic to track writes across calls
            use std::sync::atomic::{AtomicU64, Ordering};
            static WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);
            const FLUSH_INTERVAL: u64 = 100; // Flush every 100 writes

            let count = WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);

            if count % FLUSH_INTERVAL == 0 {
                // Time to flush - push memtable to SST
                if let Some(cf) = db.cf_handle("blocks") {
                    let flush_start = std::time::Instant::now();
                    if let Err(e) = db.flush_cf(&cf) {
                        tracing::warn!("⚠️  Failed to flush blocks CF: {}", e);
                    } else {
                        tracing::debug!("🔄 Batched flush #{} completed in {:?}", count / FLUSH_INTERVAL, flush_start.elapsed());
                    }
                }
            }

            tracing::debug!("💾 RocksDB write_batch #{} completed in {:?}", count, blocking_start.elapsed());

            Ok::<(), anyhow::Error>(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))??;

        debug!("✅ write_batch total time: {:?}", start.elapsed());
        Ok(())
    }

    async fn write_batch_bulk(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();

        // Prepare WriteBatch in async context (cheap)
        let mut write_batch = WriteBatch::default();

        for (cf_name, key, value) in &batch {
            let cf_handle = self.get_cf(cf_name)?;
            write_batch.put_cf(&cf_handle, key, value);
        }

        // Clone Arc for move into blocking context
        let db = self.db.clone();

        // 🚨 CRITICAL FIX v0.9.94-beta: DEADLOCK RESOLUTION
        // Even though bulk mode disables sync/WAL, write_opt() can still block
        // during compaction. Use spawn_blocking for consistency and safety.

        tokio::task::spawn_blocking(move || {
            let blocking_start = Instant::now();

            // 🚀 BULK MODE OPTIMIZATIONS - 10-100x faster for initial sync
            let mut write_opts = rocksdb::WriteOptions::default();
            write_opts.set_sync(false); // NO fsync - rely on OS page cache for speed
            write_opts.disable_wal(true); // Disable WAL for maximum write throughput

            // In bulk mode, we sacrifice durability for speed since:
            // 1. Initial sync can be restarted if it crashes
            // 2. We can re-fetch blocks from peers
            // 3. Once sync completes, we'll do a final manual flush

            db.write_opt(write_batch, &write_opts)
                .context("RocksDB bulk batch write failed")?;

            debug!("🚀 Bulk write completed in {:?} (blocking thread)", blocking_start.elapsed());

            Ok::<(), anyhow::Error>(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))??;

        debug!("✅ write_batch_bulk total time: {:?}", start.elapsed());
        Ok(())
    }

    /// 🚀 v1.0.89-beta: TURBO MODE - Maximum speed with WAL crash safety
    ///
    /// Key differences from other modes:
    /// - write_batch():      WAL=true, sync=true,  flush every 100 → ~500 BPS
    /// - write_batch_bulk(): WAL=false, sync=false, no flush      → ~2000 BPS but data loss
    /// - write_batch_turbo(): WAL=true, sync=false, no flush     → ~1500 BPS with recovery
    ///
    /// Why this works:
    /// 1. WAL enabled - on crash, RocksDB replays WAL to recover memtable state
    /// 2. sync=false - write returns immediately after copying to OS page cache
    /// 3. No flush_cf() - background compaction handles memtable → SST promotion
    /// 4. Caller calls sync_wal() periodically for durability guarantee
    ///
    /// Data loss window: From last sync_wal() call to crash
    /// Mitigation: Call sync_wal() every 1 second during sync
    async fn write_batch_turbo(&self, batch: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();
        let batch_size = batch.len();

        // Prepare WriteBatch in async context (cheap, no blocking)
        let mut write_batch = WriteBatch::default();

        for (cf_name, key, value) in &batch {
            let cf_handle = self.get_cf(cf_name)?;
            write_batch.put_cf(&cf_handle, key, value);
        }

        // Clone Arc for move into blocking context
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            let blocking_start = Instant::now();

            // 🚀 TURBO MODE: WAL enabled, sync disabled
            // - WAL captures all writes for crash recovery
            // - sync=false means no fsync() per write (massive speedup)
            // - Caller responsible for periodic sync_wal() calls
            let mut write_opts = rocksdb::WriteOptions::default();
            write_opts.set_sync(false);  // NO per-write fsync (key to speed)
            write_opts.disable_wal(false); // WAL ENABLED (key to safety)

            db.write_opt(write_batch, &write_opts)
                .context("RocksDB turbo batch write failed")?;

            tracing::debug!(
                "⚡ TURBO write: {} items in {:?} (blocking thread)",
                batch_size,
                blocking_start.elapsed()
            );

            Ok::<(), anyhow::Error>(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))??;

        debug!("⚡ write_batch_turbo total time: {:?} ({} items)", start.elapsed(), batch_size);
        Ok(())
    }

    async fn scan_prefix(&self, cf: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf_handle = self.get_cf(cf)?;
        let mut results = Vec::new();
        let iter = self.db.prefix_iterator_cf(&cf_handle, prefix);

        for item in iter {
            let (key, value) = item.context("Iterator error")?;

            // Check if key still has the prefix
            if !key.starts_with(prefix) {
                break;
            }

            results.push((key.to_vec(), value.to_vec()));
        }

        Ok(results)
    }

    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf_handle = self.get_cf(cf)?;
        let mut results = Vec::new();
        let iter = self
            .db
            .iterator_cf(&cf_handle, rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item.context("Iterator error")?;
            results.push((key.to_vec(), value.to_vec()));
        }

        Ok(results)
    }

    async fn flush(&self) -> Result<()> {
        self.db.flush().context("RocksDB flush failed")?;
        Ok(())
    }

    async fn compact(&self) -> Result<()> {
        // Use static list since cf_names() is removed in RocksDB 0.22
        let cf_names = vec![
            "default",
            "blocks",
            "dag_vertices",
            "bullshark_cert",
            "manifest",
        ];
        for cf_name in cf_names {
            if let Some(cf_handle) = self.db.cf_handle(cf_name) {
                debug!("🗜️ Compacting column family: {}", cf_name);
                self.db
                    .compact_range_cf(&cf_handle, None::<&[u8]>, None::<&[u8]>);
            }
        }
        Ok(())
    }

    async fn get_db_size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        // Get size of each column family
        let cf_names = vec!["default", "blocks", "transactions", "state", "metadata"];
        for cf_name in cf_names {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                if let Ok(Some(size_str)) = self
                    .db
                    .property_value_cf(&cf, "rocksdb.total-sst-files-size")
                {
                    if let Ok(size) = size_str.parse::<u64>() {
                        total_size += size;
                    }
                }
            }
        }

        Ok(total_size)
    }

    // 🚨 v0.9.60-beta: CRITICAL DURABILITY IMPLEMENTATIONS

    async fn create_checkpoint(&self, checkpoint_dir: &str) -> Result<()> {
        use rocksdb::checkpoint::Checkpoint;

        info!("💾 [CHECKPOINT] Creating snapshot at {}", checkpoint_dir);

        let checkpoint = Checkpoint::new(&*self.db)
            .context("Failed to create Checkpoint object")?;

        checkpoint.create_checkpoint(checkpoint_dir)
            .context("Failed to create checkpoint")?;

        info!("✅ [CHECKPOINT] Snapshot created successfully (hard-linked, zero-copy)");
        Ok(())
    }

    async fn sync_wal(&self) -> Result<()> {
        info!("🔄 [WAL SYNC] Forcing WAL to disk...");

        self.db.flush_wal(true)  // true = sync to disk
            .context("Failed to sync WAL")?;

        info!("✅ [WAL SYNC] All pending writes are now durable");
        Ok(())
    }

    async fn shutdown_gracefully(&self) -> Result<()> {
        info!("🛑 [GRACEFUL SHUTDOWN] Starting shutdown sequence...");

        // Step 1: Sync WAL
        info!("   1/3 Syncing WAL to disk...");
        self.sync_wal().await?;

        // Step 2: Flush all column families
        info!("   2/3 Flushing all column families...");

        let cf_names = vec![
            CF_BLOCKS, CF_DAG_VERTICES, CF_BULLSHARK_CERT, CF_MANIFEST,
            CF_TRANSACTIONS, CF_BALANCES, CF_BLOCK_HASH_TO_HEIGHT,
            CF_AI_CHATS, CF_AI_CREDITS, CF_AI_TRANSACTIONS, CF_AI_TREASURY,
            CF_AI_ATTACHMENTS, CF_PAYMENT_PROPOSALS, CF_PAYMENT_VOTES,
            CF_PAYMENT_LOCKS, CF_BANNED_PEERS,
        ];

        for cf_name in cf_names {
            if let Some(cf_handle) = self.db.cf_handle(cf_name) {
                self.db.flush_cf(&cf_handle)
                    .with_context(|| format!("Failed to flush CF: {}", cf_name))?;
                info!("      ✓ Flushed {}", cf_name);
            }
        }

        // Step 3: Final sync
        info!("   3/3 Final WAL sync...");
        self.db.flush_wal(true)?;

        info!("✅ [GRACEFUL SHUTDOWN] All data persisted safely. DB ready to close.");
        Ok(())
    }

    async fn verify_checkpoint(&self, checkpoint_dir: &str) -> Result<bool> {
        info!("🔍 [VERIFY] Checking checkpoint integrity at {}", checkpoint_dir);

        use std::path::Path;
        let path = Path::new(checkpoint_dir);

        // Check if checkpoint exists
        if !path.exists() {
            warn!("❌ [VERIFY] Checkpoint directory does not exist");
            return Ok(false);
        }

        // Try to open the checkpoint as a read-only database
        let mut opts = rocksdb::Options::default();
        opts.set_paranoid_checks(true);  // Maximum validation

        match rocksdb::DB::open_for_read_only(&opts, checkpoint_dir, false) {
            Ok(checkpoint_db) => {
                // Try reading manifest to verify basic integrity
                if let Some(manifest_cf) = checkpoint_db.cf_handle(CF_MANIFEST) {
                    match checkpoint_db.get_cf(&manifest_cf, b"height") {
                        Ok(_) => {
                            info!("✅ [VERIFY] Checkpoint is valid and readable");
                            Ok(true)
                        }
                        Err(e) => {
                            warn!("⚠️ [VERIFY] Checkpoint opened but read failed: {}", e);
                            Ok(false)
                        }
                    }
                } else {
                    warn!("⚠️ [VERIFY] Checkpoint missing expected column families");
                    Ok(false)
                }
            }
            Err(e) => {
                warn!("❌ [VERIFY] Failed to open checkpoint: {}", e);
                Ok(false)
            }
        }
    }

    /// 🚀 v1.0.100-beta: Multi-get for batch fetching (10-50x faster than N individual gets)
    /// Uses RocksDB's native multi_get_cf which batches disk I/O operations
    /// Returns a Vec of Option<Vec<u8>> in the same order as keys
    async fn multi_get(&self, cf: &str, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>> {
        use std::time::Instant;

        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let key_count = keys.len();

        // Clone data for spawn_blocking
        let db = self.db.clone();
        let keys_owned: Vec<Vec<u8>> = keys.to_vec();
        let cf_name = cf.to_string(); // CF handle is not Send-safe, so pass name

        // 🚀 PERFORMANCE: RocksDB multi_get batches disk I/O operations
        // Instead of N separate read syscalls, it uses a single batch read
        // Typical speedup: 10-50x for batch sizes of 100-1000 keys
        let results = tokio::task::spawn_blocking(move || {
            let blocking_start = Instant::now();

            // Get CF handle inside spawn_blocking (cf_handle is not Send-safe)
            let cf_handle = db.cf_handle(&cf_name)
                .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", cf_name))?;

            // Build key slices for multi_get_cf
            let key_slices: Vec<&[u8]> = keys_owned.iter().map(|k| k.as_slice()).collect();

            // Use RocksDB's native multi_get_cf for batched disk I/O
            let raw_results = db.multi_get_cf(
                key_slices.iter().map(|k| (&cf_handle, *k)).collect::<Vec<_>>()
            );

            // Convert to our Result type
            let mut results = Vec::with_capacity(key_slices.len());
            for result in raw_results {
                match result {
                    Ok(opt_val) => results.push(opt_val),
                    Err(e) => {
                        tracing::warn!("⚠️  [MULTI_GET] RocksDB error for one key: {}", e);
                        results.push(None); // Treat errors as missing keys
                    }
                }
            }

            tracing::debug!(
                "🚀 [MULTI_GET] Fetched {} keys in {:?} (blocking thread)",
                key_slices.len(),
                blocking_start.elapsed()
            );

            Ok::<Vec<Option<Vec<u8>>>, anyhow::Error>(results)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))??;

        let elapsed = start.elapsed();
        let found_count = results.iter().filter(|r| r.is_some()).count();

        tracing::debug!(
            "✅ [MULTI_GET] {} keys total, {} found, in {:?} ({:.0} keys/sec)",
            key_count,
            found_count,
            elapsed,
            if elapsed.as_secs_f64() > 0.0 { key_count as f64 / elapsed.as_secs_f64() } else { 0.0 }
        );

        Ok(results)
    }
}

/// RocksDB write options optimized for Narwhal workloads
#[cfg(not(target_os = "windows"))]
impl RocksDBKV {
    /// Get optimized write options
    fn write_options() -> rocksdb::WriteOptions {
        let mut opts = rocksdb::WriteOptions::default();
        opts.set_sync(true); // CRITICAL: Force fsync() to survive hard kills (pkill -9)
        opts.disable_wal(false); // Keep WAL for crash recovery
        opts
    }

    /// Get optimized read options
    fn read_options() -> rocksdb::ReadOptions {
        let mut opts = rocksdb::ReadOptions::default();
        opts.set_verify_checksums(false); // Trade off for speed in hot path
        opts
    }

    /// Write batch atomically (internal method for transactions)
    ///
    /// **SECURITY FIX (v0.8.1-beta)**: Used by QTransaction to write atomic batches
    /// **PERFORMANCE FIX (v1.0.2-beta)**: Use spawn_blocking to prevent Tokio executor thread starvation
    pub async fn write_batch_internal(
        &self,
        batch: WriteBatch,
        write_opts: rocksdb::WriteOptions,
    ) -> Result<()> {
        // ✅ v1.0.2-beta Layer 2 FIX: Move blocking RocksDB operations to dedicated thread pool
        // Prevents Tokio executor threads from being blocked by slow disk I/O
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            // Blocking RocksDB write operation (fsync to disk)
            db.write_opt(batch, &write_opts)
                .context("RocksDB batch write failed")?;

            // v1.0.77-beta: Keep flush_cf() for crash safety
            // Without explicit flushes, kill -9 causes data loss because:
            // - set_sync(false) only writes to WAL buffer
            // - WAL may not be fully synced to disk on hard kill
            // - Blocks 299900-304400 were lost due to this
            //
            // SOLUTION: Keep the flush but only on critical column families
            // and only when the batch contains important data (blocks)
            let cf_names = vec!["blocks"]; // Only flush blocks CF for safety

            for cf_name in cf_names {
                if let Some(cf) = db.cf_handle(cf_name) {
                    if let Err(e) = db.flush_cf(&cf) {
                        tracing::warn!("⚠️  Failed to flush CF {} after commit: {}", cf_name, e);
                    }
                }
            }

            Ok::<(), anyhow::Error>(())
        })
        .await
        .context("spawn_blocking task panicked")?
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> Result<RocksDBStats> {
        let mut cf_stats = HashMap::new();

        // Use static list since cf_names() is removed in RocksDB 0.22
        let cf_names = vec![
            "default",
            "blocks",
            "dag_vertices",
            "bullshark_cert",
            "manifest",
        ];
        for cf_name in cf_names {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                let stats = RocksDBCFStats {
                    keys: Self::get_cf_property(&self.db, &cf, "rocksdb.estimate-num-keys")?,
                    size: Self::get_cf_property(&self.db, &cf, "rocksdb.total-sst-files-size")?,
                    files: Self::get_cf_property(&self.db, &cf, "rocksdb.num-files-at-level0")?,
                    compactions: Self::get_cf_property(
                        &self.db,
                        &cf,
                        "rocksdb.num-running-compactions",
                    )?,
                };
                cf_stats.insert(cf_name.to_string(), stats);
            }
        }

        Ok(RocksDBStats {
            column_families: cf_stats,
            total_size: Self::get_total_size(&self.db)?,
            cache_usage: Self::get_cache_usage(&self.db)?,
        })
    }

    /// Get property value from column family
    fn get_cf_property(
        db: &DB,
        cf: &Arc<rocksdb::BoundColumnFamily>,
        property: &str,
    ) -> Result<u64> {
        db.property_value_cf(cf, property)
            .context("Failed to get property")?
            .context("Property value missing")?
            .parse()
            .context("Failed to parse property value")
    }

    /// Get total database size
    fn get_total_size(db: &DB) -> Result<u64> {
        let mut total_size = 0u64;

        let cf_names = vec!["default", "blocks", "transactions", "state", "metadata"];
        for cf_name in cf_names {
            if let Some(cf) = db.cf_handle(cf_name) {
                if let Ok(size) = Self::get_cf_property(db, &cf, "rocksdb.total-sst-files-size") {
                    total_size += size;
                }
            }
        }

        Ok(total_size)
    }

    /// Get cache usage
    fn get_cache_usage(_db: &DB) -> Result<u64> {
        // TODO: Implement cache usage tracking
        Ok(0)
    }

    /// Create checkpoint for snapshot
    pub async fn create_checkpoint<P: AsRef<Path>>(&self, checkpoint_path: P) -> Result<()> {
        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.db)
            .context("Failed to create checkpoint handle")?;

        checkpoint
            .create_checkpoint(checkpoint_path.as_ref())
            .context("Failed to create checkpoint")?;

        Ok(())
    }

    /// Execute adaptive pruning based on current configuration
    /// Deletes old blocks while preserving checkpoints and recent data
    pub async fn prune_old_blocks(&self, current_height: u64) -> Result<crate::pruning::PruningStats> {
        use crate::pruning::AdaptivePruningEngine;
        use std::time::{SystemTime, Instant};

        let start_time = Instant::now();
        info!("✂️  Starting adaptive pruning at height {}", current_height);

        // Get storage size before pruning
        let storage_before = self.get_db_size().await?;

        // Initialize pruning engine with current configuration
        let pruning_engine = AdaptivePruningEngine::new(&self.db_path, self.pruning_config.clone());

        let mut pruned_blocks = 0u64;
        let mut retained_blocks = 0u64;

        // Get column family handles
        let cf_blocks = self.get_cf(CF_BLOCKS)?;
        let cf_dag_vertices = self.get_cf(CF_DAG_VERTICES)?;
        let cf_bullshark_cert = self.get_cf(CF_BULLSHARK_CERT)?;

        // Calculate pruning range based on retention policy
        let retention_blocks = self.pruning_config.retain_recent_blocks_days * 43_200; // ~2 second block time
        let prune_up_to = current_height.saturating_sub(retention_blocks);

        info!(
            "📊 Pruning range: 0 to {} (retention: {} blocks)",
            prune_up_to, retention_blocks
        );

        // Iterate through blocks and prune based on retention policy
        // Use atomic batching every 1000 blocks for consistency
        const BATCH_SIZE: u64 = 1000;
        let mut current_batch_start = 0u64;

        while current_batch_start <= prune_up_to {
            let batch_end = std::cmp::min(current_batch_start + BATCH_SIZE - 1, prune_up_to);

            // Create atomic write batch for this chunk
            let mut batch = rocksdb::WriteBatch::default();
            let mut batch_pruned = 0u64;
            let mut batch_retained = 0u64;

            for height in current_batch_start..=batch_end {
                // Check if block should be retained (checkpoints, recent, etc.)
                match pruning_engine.should_retain_block(height, current_height) {
                    Ok(should_retain) => {
                        if !should_retain {
                            // Delete block from CF_BLOCKS
                            let block_key = height.to_be_bytes();
                            batch.delete_cf(&cf_blocks, &block_key);

                            batch_pruned += 1;

                            // Also delete associated DAG vertices (round-based)
                            // DAG vertices use (round || author || seq) as key
                            let round_prefix = height.to_be_bytes();
                            let iter = self.db.prefix_iterator_cf(&cf_dag_vertices, &round_prefix);

                            for item in iter {
                                if let Ok((key, _value)) = item {
                                    if key.starts_with(&round_prefix) {
                                        batch.delete_cf(&cf_dag_vertices, &key);
                                    } else {
                                        break; // No more vertices for this round
                                    }
                                }
                            }

                            // Delete Bullshark certificate (round -> certificate)
                            if self.db.get_cf(&cf_bullshark_cert, &block_key)?.is_some() {
                                batch.delete_cf(&cf_bullshark_cert, &block_key);
                            }
                        } else {
                            batch_retained += 1;
                        }
                    }
                    Err(e) => {
                        warn!("⚠️  Error checking retention for block {}: {}", height, e);
                        // Continue pruning other blocks even if one fails
                        batch_retained += 1; // Count as retained to be safe
                    }
                }
            }

            // Atomically commit this batch
            if batch_pruned > 0 {
                self.db.write(batch)
                    .context(format!("Failed to commit pruning batch {}-{}", current_batch_start, batch_end))?;
                debug!("✅ Pruned batch {}-{}: {} deleted, {} retained",
                       current_batch_start, batch_end, batch_pruned, batch_retained);
            }

            pruned_blocks += batch_pruned;
            retained_blocks += batch_retained;

            // Log progress every 10,000 blocks
            if batch_end % 10_000 < BATCH_SIZE && batch_end > 0 {
                info!(
                    "🗑️  Pruning progress: {} blocks checked, {} deleted, {} retained",
                    batch_end, pruned_blocks, retained_blocks
                );
            }

            current_batch_start = batch_end + 1;
        }

        // Count remaining blocks (from prune_up_to to current_height)
        retained_blocks += current_height.saturating_sub(prune_up_to);

        // Compact database after pruning to reclaim space
        info!("🗜️  Compacting database after pruning...");
        self.db.compact_range_cf(&cf_blocks, None::<&[u8]>, None::<&[u8]>);
        self.db.compact_range_cf(&cf_dag_vertices, None::<&[u8]>, None::<&[u8]>);
        self.db.compact_range_cf(&cf_bullshark_cert, None::<&[u8]>, None::<&[u8]>);

        // Get storage size after pruning
        let storage_after = self.get_db_size().await?;
        let space_saved = storage_before.saturating_sub(storage_after);

        let prune_duration_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "✅ Pruning complete: {} blocks pruned, {} blocks retained, {:.2} MB saved, took {}ms",
            pruned_blocks,
            retained_blocks,
            space_saved as f64 / 1_000_000.0,
            prune_duration_ms
        );

        Ok(crate::pruning::PruningStats {
            total_blocks: current_height,
            pruned_blocks,
            retained_blocks,
            storage_before,
            storage_after,
            space_saved,
            last_prune_time: SystemTime::now(),
            prune_duration_ms,
        })
    }

    /// Get current blockchain height from storage
    pub async fn get_blockchain_height(&self) -> Result<u64> {
        // Try to get the latest block height from manifest
        let cf_manifest = self.get_cf(CF_MANIFEST)?;

        if let Some(height_bytes) = self.db.get_cf(&cf_manifest, b"blockchain_height")? {
            if height_bytes.len() == 8 {
                let bytes: [u8; 8] = height_bytes.as_slice().try_into()
                    .map_err(|_| anyhow::anyhow!("Invalid blockchain height format"))?;
                let height = u64::from_be_bytes(bytes);
                Ok(height)
            } else {
                Err(anyhow::anyhow!("Invalid blockchain height size: expected 8 bytes, got {}", height_bytes.len()))
            }
        } else {
            // If not found in manifest, scan blocks CF to find highest
            let cf_blocks = self.get_cf(CF_BLOCKS)?;
            let mut max_height = 0u64;

            let iter = self.db.iterator_cf(&cf_blocks, rocksdb::IteratorMode::Start);
            for item in iter {
                if let Ok((key, _)) = item {
                    if key.len() >= 8 {
                        let height = u64::from_be_bytes(
                            key[0..8].try_into().unwrap_or([0u8; 8])
                        );
                        max_height = max_height.max(height);
                    }
                }
            }

            Ok(max_height)
        }
    }

    /// Set pruning configuration
    pub fn set_pruning_config(&mut self, config: crate::pruning::PruningConfig) {
        info!("⚙️  Updating pruning configuration: {:?}", config.mode);
        self.pruning_config = config;
    }

    /// Get current pruning configuration
    pub fn get_pruning_config(&self) -> &crate::pruning::PruningConfig {
        &self.pruning_config
    }

    /// Get Arc<DB> handle for SafeBatchedWriter (v1.0.2-beta Phase 1A)
    pub fn db(&self) -> Arc<DB> {
        self.db.clone()
    }

    // ==================== v1.0.43-beta: RocksDB Encryption Integration ====================

    /// Initialize encryption manager if environment variables are set
    ///
    /// Checks for:
    /// - Q_ENCRYPTION_KEYS_FILE: Path to encryption keys file
    /// - Q_ENCRYPTION_PASSPHRASE: Passphrase for key derivation
    ///
    /// BEHAVIOR:
    /// - If both variables are set: Initialize EncryptionManager (auto-generates keys if needed)
    /// - If neither is set: Returns None (encryption disabled)
    /// - If only one is set: Returns error (misconfiguration)
    ///
    /// AUTOMATIC KEY GENERATION:
    /// - If keys file doesn't exist, automatically generates new keys with ZK-STARK proof
    /// - No manual commands required!
    /// - Takes ~500ms for ZK-STARK proof generation on first run
    /// 🔐 v1.0.44-beta: MANDATORY encryption with auto-generated passphrases
    ///
    /// SECURITY: Encryption is now REQUIRED for all nodes. If not configured,
    /// a cryptographically random passphrase is auto-generated and saved.
    ///
    /// This prevents privacy leaks where unencrypted nodes expose all blockchain data.
    fn initialize_encryption_if_enabled() -> Result<Option<Arc<crate::encryption::EncryptionManager>>> {
        use std::path::PathBuf;

        // Default paths if not configured
        let keys_file_path = std::env::var("Q_ENCRYPTION_KEYS_FILE")
            .unwrap_or_else(|_| {
                let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data".to_string());
                format!("{}/encryption.keys", db_path)
            });

        // v1.0.52-beta: AUTOMATIC PASSPHRASE MANAGEMENT
        // Priority: 1) Environment variable, 2) Saved passphrase file, 3) Generate new
        let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data".to_string());
        let passphrase_file = format!("{}/encryption_passphrase.txt", db_path);

        let passphrase = if let Ok(env_pass) = std::env::var("Q_ENCRYPTION_PASSPHRASE") {
            // User explicitly set passphrase via environment
            info!("🔐 Using passphrase from Q_ENCRYPTION_PASSPHRASE environment variable");
            env_pass
        } else if std::path::Path::new(&passphrase_file).exists() {
            // AUTO-LOAD existing passphrase from file (USER-FRIENDLY!)
            match std::fs::read_to_string(&passphrase_file) {
                Ok(saved_pass) => {
                    let trimmed = saved_pass.trim().to_string();
                    info!("🔐 Auto-loaded passphrase from: {}", passphrase_file);
                    info!("   ✅ No manual configuration needed!");
                    trimmed
                }
                Err(e) => {
                    error!("❌ Failed to read existing passphrase file: {}", e);
                    error!("   File exists but is not readable: {}", passphrase_file);
                    return Err(anyhow::anyhow!(
                        "Cannot read existing passphrase file '{}': {}. \
                        Fix file permissions or delete it to generate a new database.",
                        passphrase_file, e
                    ));
                }
            }
        } else {
            // Generate NEW passphrase for fresh database
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
            let random_pass = random_bytes.iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();

            info!("🔐 Generating new encryption passphrase (first run)");

            // Ensure data directory exists before saving
            if let Err(e) = std::fs::create_dir_all(&db_path) {
                warn!("⚠️  Could not create data directory {}: {}", db_path, e);
            }

            match std::fs::write(&passphrase_file, &random_pass) {
                Ok(_) => {
                    info!("💾 Passphrase saved to: {}", passphrase_file);
                    info!("   ✅ Will be auto-loaded on next startup - no manual config needed!");
                    info!("   ⚠️  BACKUP THIS FILE if you want to recover data after disk failure.");
                }
                Err(e) => {
                    error!("❌ Failed to save passphrase file: {}", e);
                    error!("   Your passphrase: {}", random_pass);
                    error!("   SAVE THIS IMMEDIATELY or you'll lose access to your database!");
                }
            }

            random_pass
        };

        info!("🔐 Initializing MANDATORY encryption (v1.0.44-beta)");
        info!("   Keys file: {}", keys_file_path);
        info!("   Passphrase: {}...", if passphrase.len() > 8 { &passphrase[..8] } else { &passphrase });
        info!("   Encryption with ZK-STARK untrusted setup...");

        let keys_file = PathBuf::from(keys_file_path);

        // This will auto-generate keys if they don't exist!
        let encryption_manager = crate::encryption::EncryptionManager::from_passphrase(
            &passphrase,
            &keys_file
        )?;

        info!("✅ Encryption manager initialized successfully");
        info!("   Database encryption is now ACTIVE (MANDATORY)");
        info!("   All data written to RocksDB is encrypted with AES-256-GCM");
        info!("   Old unencrypted data migrates during compaction");

        Ok(Some(Arc::new(encryption_manager)))
    }

    /// Get encryption manager (if enabled)
    pub fn get_encryption_manager(&self) -> Option<Arc<crate::encryption::EncryptionManager>> {
        self.encryption_manager.clone()
    }

    // ==================== End RocksDB Encryption Integration ====================

    // ==================== Phase 2: Block-Vertex Mapping (v1.0.4-beta) ====================
    // These methods provide persistent storage for BlockVertexMap
    // Used by DAG-aware sync for 20-40x performance improvement

    /// Store block hash → vertex ID mapping
    /// Key format: "bv:{block_hash}" → vertex_id (u64 big-endian)
    pub async fn store_block_vertex_mapping(
        &self,
        block_hash: &str,
        vertex_id: u64,
    ) -> Result<()> {
        let key = format!("bv:{}", block_hash);
        let value = vertex_id.to_be_bytes(); // Big-endian for consistent ordering

        let db = self.db.clone();
        let key_bytes = key.into_bytes();
        let value_bytes = value.to_vec();

        tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;
            db.put_cf(&cf, &key_bytes, &value_bytes)
                .context("Failed to store block-vertex mapping")
        })
        .await??;

        Ok(())
    }

    /// Get vertex ID for block hash
    /// Returns None if block hash not found in mapping
    pub async fn get_vertex_for_block(&self, block_hash: &str) -> Result<Option<u64>> {
        let key = format!("bv:{}", block_hash);

        let db = self.db.clone();
        let key_bytes = key.into_bytes();

        let value_opt = tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;
            db.get_cf(&cf, &key_bytes)
                .context("Failed to get block-vertex mapping")
        })
        .await??;

        match value_opt {
            Some(bytes) if bytes.len() == 8 => {
                let vertex_id = u64::from_be_bytes(
                    bytes.as_slice().try_into()
                        .map_err(|_| anyhow::anyhow!("Invalid vertex_id bytes"))?
                );
                Ok(Some(vertex_id))
            }
            Some(_) => Err(anyhow::anyhow!("Invalid vertex_id size: expected 8 bytes")),
            None => Ok(None),
        }
    }

    /// Store vertex ID → block hash mapping (reverse index)
    /// Key format: "vb:{vertex_id}" → block_hash (UTF-8 string)
    pub async fn store_vertex_block_mapping(
        &self,
        vertex_id: u64,
        block_hash: &str,
    ) -> Result<()> {
        let key = format!("vb:{}", vertex_id);
        let value = block_hash.as_bytes();

        let db = self.db.clone();
        let key_bytes = key.into_bytes();
        let value_bytes = value.to_vec();

        tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;
            db.put_cf(&cf, &key_bytes, &value_bytes)
                .context("Failed to store vertex-block mapping")
        })
        .await??;

        Ok(())
    }

    /// Get block hash for vertex ID
    /// Returns None if vertex ID not found in mapping
    pub async fn get_block_for_vertex(&self, vertex_id: u64) -> Result<Option<String>> {
        let key = format!("vb:{}", vertex_id);

        let db = self.db.clone();
        let key_bytes = key.into_bytes();

        let value_opt = tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;
            db.get_cf(&cf, &key_bytes)
                .context("Failed to get vertex-block mapping")
        })
        .await??;

        match value_opt {
            Some(bytes) => {
                let block_hash = String::from_utf8(bytes)
                    .context("Invalid UTF-8 in block hash")?;
                Ok(Some(block_hash))
            }
            None => Ok(None),
        }
    }

    /// Batch store block-vertex mappings (optimized for sync)
    /// Stores both directions: block→vertex and vertex→block
    /// This is CRITICAL for Phase 2 sync performance (atomic batch writes)
    pub async fn batch_store_mappings(
        &self,
        mappings: &[(String, u64)], // (block_hash, vertex_id) pairs
    ) -> Result<()> {
        if mappings.is_empty() {
            return Ok(());
        }

        let db = self.db.clone();

        // Clone data for spawn_blocking
        let mappings_owned: Vec<(String, u64)> = mappings.to_vec();

        tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;

            let mut batch = WriteBatch::default();

            for (block_hash, vertex_id) in mappings_owned {
                // Store block → vertex mapping
                let bv_key = format!("bv:{}", block_hash);
                let bv_value = vertex_id.to_be_bytes();
                batch.put_cf(&cf, bv_key.as_bytes(), &bv_value);

                // Store vertex → block mapping (reverse index)
                let vb_key = format!("vb:{}", vertex_id);
                batch.put_cf(&cf, vb_key.as_bytes(), block_hash.as_bytes());
            }

            // Write entire batch atomically
            db.write(batch)
                .context("Failed to batch store block-vertex mappings")
        })
        .await??;

        debug!("✅ Batch stored {} block-vertex mappings", mappings.len());
        Ok(())
    }

    /// Delete block-vertex mappings (for testing/cleanup)
    pub async fn delete_block_vertex_mapping(&self, block_hash: &str, vertex_id: u64) -> Result<()> {
        let db = self.db.clone();

        let bv_key = format!("bv:{}", block_hash);
        let vb_key = format!("vb:{}", vertex_id);

        tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;

            let mut batch = WriteBatch::default();
            batch.delete_cf(&cf, bv_key.as_bytes());
            batch.delete_cf(&cf, vb_key.as_bytes());

            db.write(batch)
                .context("Failed to delete block-vertex mappings")
        })
        .await??;

        Ok(())
    }

    /// Get all block-vertex mappings (for migration/debugging)
    /// WARNING: This can be slow for large blockchains
    pub async fn get_all_block_vertex_mappings(&self) -> Result<Vec<(String, u64)>> {
        let db = self.db.clone();

        let mappings = tokio::task::spawn_blocking(move || {
            let cf = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF not found: {}", CF_MANIFEST))?;

            let prefix = b"bv:";
            let mut result = Vec::new();

            let iter = db.prefix_iterator_cf(&cf, prefix);
            for item in iter {
                if let Ok((key, value)) = item {
                    // Extract block hash from key
                    if key.starts_with(prefix) && value.len() == 8 {
                        let block_hash = String::from_utf8(key[3..].to_vec())
                            .context("Invalid UTF-8 in block hash")?;
                        let vertex_id = u64::from_be_bytes(
                            value.as_ref().try_into()
                                .map_err(|_| anyhow::anyhow!("Invalid vertex_id bytes"))?
                        );
                        result.push((block_hash, vertex_id));
                    }
                }
            }

            Ok::<Vec<(String, u64)>, anyhow::Error>(result)
        })
        .await??;

        Ok(mappings)
    }

    // ==================== End Phase 2 Block-Vertex Mapping ====================
}

/// RocksDB statistics for monitoring
#[cfg(not(target_os = "windows"))]
#[derive(Debug, Clone)]
pub struct RocksDBStats {
    pub column_families: HashMap<String, RocksDBCFStats>,
    pub total_size: u64,
    pub cache_usage: u64,
}

/// Column family statistics
#[cfg(not(target_os = "windows"))]
#[derive(Debug, Clone)]
pub struct RocksDBCFStats {
    pub keys: u64,
    pub size: u64,
    pub files: u64,
    pub compactions: u64,
}

#[cfg(not(target_os = "windows"))]
impl RocksDBStats {
    /// Get Prometheus-format metrics
    pub fn to_prometheus(&self) -> String {
        let mut metrics = String::new();

        for (cf_name, stats) in &self.column_families {
            metrics.push_str(&format!(
                "rocksdb_keys{{cf=\"{}\"}} {}\n\
                 rocksdb_size_bytes{{cf=\"{}\"}} {}\n\
                 rocksdb_files{{cf=\"{}\"}} {}\n\
                 rocksdb_compactions{{cf=\"{}\"}} {}\n",
                cf_name,
                stats.keys,
                cf_name,
                stats.size,
                cf_name,
                stats.files,
                cf_name,
                stats.compactions
            ));
        }

        metrics.push_str(&format!(
            "rocksdb_total_size_bytes {}\n\
             rocksdb_cache_usage_bytes {}\n",
            self.total_size, self.cache_usage
        ));

        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_hot_db_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = RocksDBKV::open_hot_db(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cold_db_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = RocksDBKV::open_cold_db(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        // Test put/get
        let key = b"test_key";
        let value = b"test_value";

        kv.put(CF_MANIFEST, key, value).await.unwrap();
        let retrieved = kv.get(CF_MANIFEST, key).await.unwrap();

        assert_eq!(retrieved, Some(value.to_vec()));

        // Test delete
        kv.delete(CF_MANIFEST, key).await.unwrap();
        let after_delete = kv.get(CF_MANIFEST, key).await.unwrap();

        assert_eq!(after_delete, None);
    }

    #[tokio::test]
    async fn test_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        let batch = vec![
            (CF_MANIFEST, b"key1".to_vec(), b"value1".to_vec()),
            (CF_MANIFEST, b"key2".to_vec(), b"value2".to_vec()),
            (CF_MANIFEST, b"key3".to_vec(), b"value3".to_vec()),
        ];

        kv.write_batch(batch).await.unwrap();

        // Verify all keys were written
        assert_eq!(
            kv.get(CF_MANIFEST, b"key1").await.unwrap(),
            Some(b"value1".to_vec())
        );
        assert_eq!(
            kv.get(CF_MANIFEST, b"key2").await.unwrap(),
            Some(b"value2".to_vec())
        );
        assert_eq!(
            kv.get(CF_MANIFEST, b"key3").await.unwrap(),
            Some(b"value3".to_vec())
        );
    }

    #[tokio::test]
    async fn test_prefix_scan() {
        let temp_dir = TempDir::new().unwrap();
        let kv = RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap();

        // Insert test data with common prefix
        let prefix = b"test_prefix_";
        for i in 0..5 {
            let key = format!("{}key{}", std::str::from_utf8(prefix).unwrap(), i);
            let value = format!("value{}", i);
            kv.put(CF_MANIFEST, key.as_bytes(), value.as_bytes())
                .await
                .unwrap();
        }

        // Scan with prefix
        let results = kv.scan_prefix(CF_MANIFEST, prefix).await.unwrap();
        assert_eq!(results.len(), 5);

        // Verify results are sorted
        for (i, (key, _)) in results.iter().enumerate() {
            let expected_key = format!("{}key{}", std::str::from_utf8(prefix).unwrap(), i);
            assert_eq!(key, expected_key.as_bytes());
        }
    }
}
