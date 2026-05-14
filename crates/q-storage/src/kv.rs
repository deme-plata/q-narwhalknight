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
    CF_BLOCK_HASH_TO_HEIGHT, CF_BLOCKS, CF_BULLSHARK_CERT, CF_CALENDAR_BY_DATE, CF_CALENDAR_COMMUNITY,
    CF_CALENDAR_EVENTS, CF_CALENDAR_SCHEDULED_TX, CF_DAG_VERTICES, CF_MANIFEST,
    CF_NARWHAL_PAYLOADS, CF_PAYMENT_LOCKS, CF_PAYMENT_PROPOSALS, CF_PAYMENT_VOTES, CF_TRANSACTIONS,
};

/// Parse height from DAG key bytes without allocation.
/// Format: b"qblock:dag:{height}:{proposer}"
#[inline]
fn parse_dag_key_height(key: &[u8]) -> Option<u64> {
    const PREFIX: &[u8] = b"qblock:dag:";
    const MAX_HEIGHT: u64 = 1_000_000_000; // ~30,000 years at 1 bps
    if !key.starts_with(PREFIX) { return None; }
    let mut pos = PREFIX.len();
    let mut height: u64 = 0;
    while pos < key.len() {
        match key[pos] {
            b':' => {
                if height > MAX_HEIGHT { return None; }
                return Some(height);
            }
            b'0'..=b'9' => {
                height = height.checked_mul(10)?.checked_add((key[pos] - b'0') as u64)?;
                pos += 1;
            }
            _ => return None,
        }
    }
    None
}

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

    /// Seek-based prefix scan — works WITHOUT a prefix extractor on the CF.
    /// Uses raw iterator seek (B-tree traversal) instead of prefix_iterator_cf
    /// (which relies on bloom filters that give false negatives without extractor).
    /// v10.3.7: Created to fix 545K invisible DAG blocks on CF_BLOCKS.
    async fn scan_prefix_seek(&self, cf: &str, prefix: &[u8], limit: usize) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Scan all keys in column family (use with caution)
    async fn scan_all(&self, cf: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// v10.3.7: Forward iterate DAG blocks from start_height using lazy iterator.
    /// Returns (height, key_bytes, value_bytes) tuples sorted by numeric height.
    /// Default: returns empty (sled/Windows). RocksDB impl uses raw iterator.
    async fn get_dag_blocks_forward(
        &self,
        _start_height: u64,
        _limit: usize,
    ) -> Result<Vec<(u64, Vec<u8>, Vec<u8>)>> {
        Ok(vec![])
    }

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

        // v6.0.3: Auto-detect system RAM and adapt all memory-sensitive settings
        // Prevents OOM kills on low-memory systems (e.g., Gamma with 7.8 GB)
        let total_ram_mb = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            (sys.total_memory() / (1024 * 1024)) as usize  // bytes → MB
        };

        // RAM tier determines all memory-sensitive defaults
        // v8.6.0: Added "xxlarge" tier for 64GB+ systems
        let ram_tier = match total_ram_mb {
            0..=3999     => "micro",    // ≤4 GB: bare minimum
            4000..=7999  => "small",    // 4-8 GB: Gamma-class (7.8 GB)
            8000..=15999 => "medium",   // 8-16 GB: comfortable
            16000..=31999 => "large",   // 16-32 GB: recommended
            32000..=63999 => "xlarge",  // 32-64 GB: power user
            _            => "xxlarge",  // 64+ GB: high-memory (Beta 94 GB)
        };

        // Auto-scale block cache: 2-10% of RAM depending on tier
        // v6.1.0: Reduced small tier to 128MB fixed to prevent OOM on Gamma (7.8GB)
        // v9.2.0: Reduced medium from 10%→5% to fix OOM on 16GB during sync
        // v10.0.9: MAJOR REDUCTION — users with 8-64GB RAM hitting OOM during sync.
        //   Root cause: 30-35% of RAM for block cache left too little for Tor (8GB),
        //   jemalloc fragmentation, kernel buffers, and sync working set.
        //   64GB machine: old=22.4GB cache, new=4GB. Total node RAM: ~8GB (vs ~35GB).
        //   Block cache hit rate is still >95% at 2-4GB for our 10M-block chain.
        let auto_cache_mb = match ram_tier {
            "micro"  => 64,                                               // 64 MB fixed
            "small"  => 128,                                              // 128 MB fixed (was 256, OOM fix)
            "medium" => (total_ram_mb * 5 / 100).clamp(256, 512),        // 5% of RAM, 256-512 MB
            "large"  => (total_ram_mb * 10 / 100).clamp(1024, 2048),    // v10.0.9: 10% of RAM, 1-2 GB (was 25%, 2-8 GB)
            "xlarge" => (total_ram_mb * 8 / 100).clamp(2048, 4096),     // v10.0.9: 8% of RAM, 2-4 GB (was 30%, 4-16 GB)
            _        => (total_ram_mb * 6 / 100).clamp(2048, 4096),     // v10.0.9: 6% of RAM, 2-4 GB (was 35%, 8-24 GB)
        };

        // Auto-scale write buffer size (DB-level default CF)
        // v6.1.0: Reduced small tier from 32→16MB (47 CFs × write_buf = too much)
        // v9.0.7: Reduced medium tier from 64→32MB to fix OOM on 16GB nodes during sync
        let auto_write_buffer_mb = match ram_tier {
            "micro"  => 8,
            "small"  => 16,
            "medium" => 32,   // v9.0.7: was 64, OOM on 16GB during sync
            _        => 128,
        };

        // Auto-scale write buffer count
        // v6.0.4: Reduced small tier from 4 to 2 to prevent OOM
        let auto_write_buffer_count = match ram_tier {
            "micro"  => 2,
            "small"  => 2,
            "medium" => 4,
            _        => 6,
        };

        info!(
            "🧠 [RAM AUTO-TUNE] Detected {} MB total RAM → tier={}, block_cache={}MB, write_buf={}MB×{}",
            total_ram_mb, ram_tier, auto_cache_mb, auto_write_buffer_mb, auto_write_buffer_count
        );

        // v6.1.0: Auto-scale RocksDB background threads
        // With 47 CFs, need enough flush threads to drain memtables during burst writes.
        // Minimum 4 bg_jobs / 2 flushes even on 2-core systems to prevent memtable pile-up.
        let num_cores = num_cpus::get();
        // v9.1.6: Raised ceilings for high-core-count machines (e.g. Epsilon 48-core).
        // 48 cores → bg_jobs=6, compactions=3, flushes=2 (auto) — override via env vars for more.
        let auto_bg_jobs = (num_cores / 8).clamp(4, 24) as i32;
        let auto_compactions = (num_cores / 16).clamp(2, 12) as i32;
        let auto_flushes = (num_cores / 32).clamp(2, 6) as i32;

        let bg_jobs = std::env::var("ROCKSDB_MAX_BACKGROUND_JOBS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(auto_bg_jobs);
        let bg_compactions = std::env::var("ROCKSDB_MAX_COMPACTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(auto_compactions);
        let bg_flushes = std::env::var("ROCKSDB_MAX_FLUSHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(auto_flushes);

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

        // v6.0.3: RAM-aware cache sizing — env var overrides auto-detection
        let cache_size_bytes = std::env::var("ROCKSDB_BLOCK_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(auto_cache_mb) * 1024 * 1024;
        let block_cache = rocksdb::Cache::new_lru_cache(cache_size_bytes);
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_cache(&block_cache);
        block_opts.set_bloom_filter(10.0, true);
        opts.set_block_based_table_factory(&block_opts);

        info!("🗄️ RocksDB block cache: {} MB (auto={} MB, tier={})",
              cache_size_bytes / (1024 * 1024), auto_cache_mb, ram_tier);

        if turbo_sync_mode {
            info!("🚀 TURBO SYNC MODE ENABLED - RAM-aware bulk write optimization");
            // v6.0.4: Scale turbo sync buffers to available RAM (reduced for OOM safety)
            let turbo_write_buf = match ram_tier {
                "micro"  => 32,    // 32 MB — very conservative
                "small"  => 64,    // 64 MB — safe for 8 GB (was 128)
                "medium" => 128,   // 128 MB
                _        => 256,   // 256 MB — big servers (was 512)
            };
            let turbo_write_count = match ram_tier {
                "micro"  => 2,
                "small"  => 2,     // was 4
                _        => 4,     // was 8
            };
            opts.set_write_buffer_size(turbo_write_buf * 1024 * 1024);
            opts.set_max_write_buffer_number(turbo_write_count);
            opts.set_target_file_size_base(256 * 1024 * 1024);
            opts.set_level_zero_file_num_compaction_trigger(16);
            opts.set_level_zero_slowdown_writes_trigger(32);
            opts.set_level_zero_stop_writes_trigger(64);

            info!("🚀 Turbo write buffers: {}MB × {} (tier={})",
                  turbo_write_buf, turbo_write_count, ram_tier);
        } else {
            // Normal mode: RAM-aware write buffers
            opts.set_write_buffer_size(auto_write_buffer_mb * 1024 * 1024);
            opts.set_max_write_buffer_number(auto_write_buffer_count);
            opts.set_target_file_size_base(128 * 1024 * 1024);
            opts.set_level_zero_file_num_compaction_trigger(2);
            opts.set_level_zero_slowdown_writes_trigger(6);
            opts.set_level_zero_stop_writes_trigger(10);
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
        opts.set_max_total_wal_size(128 * 1024 * 1024); // v8.6.0: 128MB total WAL budget (was 64MB — larger budget reduces WAL rotation stalls)
        // ChatGPT P0: DO NOT set manual_wal_flush(true) - auto flush is safer
        // opts.set_manual_wal_flush(true); // DISABLED per ChatGPT recommendation

        // ========== STEADY IO (PREVENT BURST CORRUPTION) ==========
        opts.set_bytes_per_sync(4 * 1024 * 1024); // v8.6.0: 4 MiB - sync data in steady chunks (was 1 MiB)
        opts.set_wal_bytes_per_sync(4 * 1024 * 1024); // v8.6.0: 4 MiB - sync WAL in steady chunks (was 1 MiB)

        // v8.4.4: RocksDB write rate limiter — prevents sync writes from monopolizing disk I/O.
        // During bulk sync, compaction/flush writes can saturate the disk, stalling API reads.
        // This limits background write throughput so read latency stays low.
        // Only affects compaction and flush writes — reads go through block cache unaffected.
        // Default: 200 MB/s (plenty for sync, but prevents 100% disk saturation).
        let write_rate_mb: i64 = std::env::var("Q_ROCKSDB_WRITE_RATE_MB")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(200);
        if write_rate_mb > 0 {
            let rate_bytes_per_sec = write_rate_mb * 1024 * 1024;
            opts.set_ratelimiter(rate_bytes_per_sec, 100_000, 10); // 100ms refill, fairness=10
            info!("🚀 [v8.4.4] RocksDB write rate limiter: {}MB/s (prevents disk saturation during sync)", write_rate_mb);
        }

        // v6.0.8: Direct I/O on small nodes to eliminate kernel page cache bloat
        // ROOT CAUSE of Gamma OOM: 9.7GB database → kernel caches SST file pages → 2-3GB page cache
        // Page cache is counted against cgroup MemoryMax, pushing total past 7GB limit.
        // Direct I/O bypasses page cache entirely:
        //   - Reads go through RocksDB's 256MB block cache (bounded)
        //   - Writes go directly to disk (no page cache doubling)
        //   - Memory savings: ~2-3GB on a 10GB database
        // Trade-off: Reads not in block cache are slower (disk I/O), but Gamma is a backup node.
        // v9.0.7: Enable direct I/O for medium tier too — page cache was consuming
        // 3-5 GB on 16GB nodes during turbo sync, causing OOM kills
        // v10.0.9: Enable direct I/O on all tiers up to "large" (≤32GB).
        // On xlarge/xxlarge (32GB+), kernel page cache has enough headroom.
        // On ≤32GB nodes, page cache competes with RocksDB+Tor for RAM → OOM.
        if ram_tier == "micro" || ram_tier == "small" || ram_tier == "medium" || ram_tier == "large" {
            opts.set_use_direct_reads(true);
            opts.set_use_direct_io_for_flush_and_compaction(true);
            info!("🔧 Direct I/O enabled for reads+compaction (eliminates page cache bloat on ≤32GB nodes)");
        }

        // ========== MEMORY BUDGET (FORCE FLUSHES) ==========
        // v6.1.0: RAM-aware memtable budget — TOTAL across all 47 CFs
        // When total memtable usage exceeds this, RocksDB triggers flushes.
        // Must be low enough to prevent OOM during burst writes (block catchup).
        // v9.0.7: Reduced medium tier memtable budget from 256→128MB for OOM safety on 16GB
        // v10.0.9: Reduced large tier memtable from 512→256MB. With 47 CFs,
        // 512MB memtable + 128MB×6 write buffers = potential 1.3GB in memtables alone.
        let memtable_budget_mb = match ram_tier {
            "micro"  => 32,   // 32MB — very tight
            "small"  => 64,   // 64MB — forces aggressive flushing (was 128, OOM)
            "medium" => 128,  // v9.0.7: 128MB (was 256, OOM on 16GB during sync)
            "large"  => 256,  // v10.0.9: 256MB (was 512, OOM reports on 16-32GB nodes)
            _        => 512,  // 512MB — for xlarge/xxlarge (32GB+)
        };
        opts.set_db_write_buffer_size(memtable_budget_mb * 1024 * 1024);
        info!("🗄️ RocksDB memtable budget: {}MB (tier={})", memtable_budget_mb, ram_tier);

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
            Self::create_blocks_cf(&block_cache),
            Self::create_dag_vertices_cf(&block_cache),
            Self::create_bullshark_cert_cf(&block_cache),
            Self::create_manifest_cf(&block_cache),
            Self::create_transactions_cf(&block_cache),
            Self::create_balances_cf(&block_cache),  // v0.8.2-beta: Balance consensus storage
            Self::create_block_hash_to_height_cf(&block_cache),  // v0.8.3-beta: Block hash index
            Self::create_ai_chats_cf(&block_cache),
            Self::create_ai_credits_cf(&block_cache),
            Self::create_ai_transactions_cf(&block_cache),
            Self::create_ai_treasury_cf(&block_cache),
            Self::create_ai_attachments_cf(&block_cache),  // v0.9.9-beta: AI chat attachments
            Self::create_payment_proposals_cf(&block_cache),
            Self::create_payment_votes_cf(&block_cache),
            Self::create_payment_locks_cf(&block_cache),
            Self::create_banned_peers_cf(&block_cache),  // v0.9.7-beta: ZK proof ban persistence
            Self::create_sync_certificates_cf(&block_cache),  // v0.9.18-beta: TurboSync AEGIS-QL certificates
            Self::create_peer_trust_cf(&block_cache),  // v0.9.18-beta: AEGIS-QL peer trust metrics
            Self::create_processed_updates_cf(&block_cache),  // ✅ v0.9.98-beta: P2P durability idempotency tracking
            // ========== v1.0.60-beta: Comprehensive State Sync CFs ==========
            Self::create_state_trie_cf(&block_cache),        // Sparse Merkle trie for state roots
            Self::create_token_balances_cf(&block_cache),    // Token balances (all tokens including QUG/QUGUSD)
            Self::create_tokens_cf(&block_cache),            // Token metadata
            Self::create_dex_pools_cf(&block_cache),         // DEX liquidity pools
            Self::create_lp_balances_cf(&block_cache),       // LP token balances
            Self::create_vaults_cf(&block_cache),            // Collateral vaults
            Self::create_oracle_prices_cf(&block_cache),     // Oracle price feeds
            Self::create_contracts_cf(&block_cache),         // Smart contracts
            Self::create_contract_storage_cf(&block_cache),  // Contract storage
            Self::create_ai_credits_v2_cf(&block_cache),     // AI credits v2 (with stats)
            Self::create_stakes_cf(&block_cache),            // Staking positions
            Self::create_nonces_cf(&block_cache),            // Account nonces
            // ========== v1.4.2-beta: QNO Prediction Staking ==========
            Self::create_qno_stakes_cf(&block_cache),        // QNO staking positions
            Self::create_qno_domains_cf(&block_cache),       // QNO prediction domains
            Self::create_qno_stats_cf(&block_cache),         // QNO global statistics
            // ========== v2.3.9-beta: Swap History ==========
            Self::create_swap_history_cf(&block_cache),      // Swap/transaction history for Token Details Modal
            // ========== v2.7.9-beta: Perpetual Trading ==========
            Self::create_perp_positions_cf(&block_cache),    // Perpetual futures positions
            Self::create_perp_orders_cf(&block_cache),       // Perpetual futures orders
            Self::create_perp_trades_cf(&block_cache),       // Perpetual futures trade history
            Self::create_perp_funding_cf(&block_cache),      // Perpetual funding rate history
            Self::create_perp_liquidations_cf(&block_cache), // Perpetual liquidation records
            // ========== v3.5.8-beta: Wallet Transaction History ==========
            Self::create_wallet_tx_index_cf(&block_cache),   // Wallet-indexed transaction history
            Self::create_wallet_swap_index_cf(&block_cache), // Wallet-indexed DEX swap history
            // ========== v3.6.0-beta: Price History ==========
            Self::create_price_history_cf(&block_cache),     // Consensus-verified price history
            // ========== v3.9.1-beta: Bank Messaging & Identity ==========
            Self::create_bank_messages_cf(&block_cache),     // Bank messages storage
            Self::create_bank_msg_index_cf(&block_cache),    // Bank message index by wallet
            Self::create_user_identities_cf(&block_cache),   // User identity records
            Self::create_death_certificates_cf(&block_cache), // Death certificates for inheritance
            // ========== v7.3.1: Block Storage Optimization ==========
            Self::create_quantum_metadata_cf(&block_cache),  // Quantum metadata (lazy-loaded from blocks)
            // ========== v7.3.2: Quillon Mail ==========
            Self::create_emails_cf(&block_cache),
            Self::create_emails_by_wallet_cf(&block_cache),
            Self::create_emails_by_folder_cf(&block_cache),
            Self::create_email_contacts_cf(&block_cache),
            Self::create_email_outbound_cf(&block_cache),
            // ========== v7.3.3: Blockchain Calendar ==========
            Self::create_calendar_events_cf(&block_cache),
            Self::create_calendar_by_date_cf(&block_cache),
            Self::create_calendar_scheduled_tx_cf(&block_cache),
            Self::create_calendar_community_cf(&block_cache),
            // ========== v10.9.23: balance_root_v2 — Sparse Merkle Tree ==========
            // CF for the SMT internal nodes + root pointer used by
            // `crates/q-storage/src/balance_smt.rs::BalanceSmt`. Added to the
            // hot DB so SMT updates can be batched atomically with wallet
            // balance writes (see crates/q-storage/src/balance_smt.rs).
            Self::create_balance_smt_cf(&block_cache),
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

        // v5.1.0: Auto-scale cold DB threads (lower than hot, cold is less active)
        let num_cores = num_cpus::get();
        let cold_auto_jobs = (num_cores / 32).clamp(1, 8) as i32;
        let cold_auto_compactions = (num_cores / 64).clamp(1, 4) as i32;
        let cold_auto_flushes = (num_cores / 128).clamp(1, 2) as i32;

        let cold_bg_jobs = std::env::var("ROCKSDB_COLD_MAX_BACKGROUND_JOBS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cold_auto_jobs);
        let cold_bg_compactions = std::env::var("ROCKSDB_COLD_MAX_COMPACTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cold_auto_compactions);
        let cold_bg_flushes = std::env::var("ROCKSDB_COLD_MAX_FLUSHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(cold_auto_flushes);

        info!(
            "🧊 RocksDB Cold DB Threading: jobs={}, compactions={}, flushes={}",
            cold_bg_jobs, cold_bg_compactions, cold_bg_flushes
        );

        opts.set_max_background_jobs(cold_bg_jobs);
        opts.set_max_background_compactions(cold_bg_compactions);
        opts.set_max_background_flushes(cold_bg_flushes);

        // v6.0.3: RAM-aware cold DB write buffers
        let cold_total_ram_mb = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            (sys.total_memory() / (1024 * 1024)) as usize
        };
        let cold_write_buf = if cold_total_ram_mb < 8000 { 32 } else { 128 };
        opts.set_write_buffer_size(cold_write_buf * 1024 * 1024);
        opts.set_max_write_buffer_number(2);
        opts.set_target_file_size_base(256 * 1024 * 1024); // 256MB
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // 🚨 v0.9.60-beta: COLD DB DURABILITY (same as hot DB)
        opts.set_use_fsync(true);
        opts.set_paranoid_checks(true);
        opts.set_atomic_flush(true);
        opts.set_bytes_per_sync(1024 * 1024); // 1 MiB
        opts.set_wal_bytes_per_sync(1024 * 1024); // 1 MiB

        // v6.0.5: Cold DB also needs a shared block cache for its CFs
        let cold_cache = rocksdb::Cache::new_lru_cache(64 * 1024 * 1024); // 64MB for cold DB
        let cfs = vec![Self::create_narwhal_payloads_cf(&cold_cache)];

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
            // v7.1.2: RocksDB requires ALL existing CFs to be opened.
            // If the DB has extra CFs not in our requested list (e.g., from a newer binary),
            // add them with default options to prevent "Column families not opened" error.
            let extra_cfs: Vec<String> = existing_cfs.iter()
                .filter(|name| !requested_cf_names.contains(name))
                .cloned()
                .collect();

            let mut all_cfs = cfs;
            if !extra_cfs.is_empty() {
                warn!("⚠️  [CF-COMPAT] DB has {} extra column families not in code: {:?}", extra_cfs.len(), extra_cfs);
                info!("🔧 [CF-COMPAT] Adding them to open list with default options");
                for cf_name in &extra_cfs {
                    let mut cf_opts = Options::default();
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                    all_cfs.push(ColumnFamilyDescriptor::new(cf_name.as_str(), cf_opts));
                }
            }

            // Normal path - database is new or already has all CFs
            let db = DB::open_cf_descriptors(&opts, &path, all_cfs).context("Failed to open RocksDB")?;

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

    /// v6.0.5: Apply shared block cache to CF options (prevents 48× separate 8MB caches)
    fn apply_shared_block_cache(opts: &mut Options, cache: &rocksdb::Cache) {
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        block_opts.set_bloom_filter(10.0, true);
        // v6.0.8: CRITICAL - force index/filter blocks INTO the shared cache
        // Without this, RocksDB stores them in unbounded separate memory.
        // With 9.7GB data across 49 CFs, this can consume 1-3GB uncapped.
        block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&block_opts);
    }

    /// v6.0.8: Scale per-CF write buffer size based on available RAM
    /// Uses a cached static to avoid calling sysinfo::System::new() 40+ times.
    fn scale_write_buffer(requested_mb: usize) -> usize {
        use std::sync::OnceLock;
        static RAM_MB: OnceLock<usize> = OnceLock::new();
        let ram_mb = *RAM_MB.get_or_init(|| {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            (sys.total_memory() / (1024 * 1024)) as usize
        });
        let scaled = match ram_mb {
            0..=3999     => requested_mb.min(2),    // micro: cap at 2MB per CF
            4000..=7999  => requested_mb.min(4),    // small (Gamma): cap at 4MB per CF
            8000..=15999 => requested_mb.min(16),   // medium: cap at 16MB per CF
            _            => requested_mb,            // large: use requested size
        };
        scaled * 1024 * 1024
    }

    /// Create blocks column family (height || hash -> block)
    fn create_blocks_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        // v7.3.5: RocksDB-level LZ4 compression on CF_BLOCKS.
        // App-level LZ4 was removed (blocks stored as QRAW) due to lz4::block
        // compress/decompress roundtrip failures. RocksDB LZ4 is transparent
        // and handles compression/decompression correctly.
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        Self::apply_shared_block_cache(&mut opts, cache);

        // 🚨 CRITICAL FIX v0.7.3-beta: Force persistent flushes (Expert-reviewed)
        // Root cause: avoid_flush_during_shutdown=true + unlimited WAL = data loss
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB - balanced (~1600 blocks/flush)
        opts.set_min_write_buffer_number_to_merge(1); // Flush immediately
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput (was 2)
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
    fn create_dag_vertices_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);

        // Enable prefix seek for round-based queries
        // Use fixed prefix length of 8 bytes for round number
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8));

        Self::apply_shared_block_cache(&mut opts, cache);

        ColumnFamilyDescriptor::new(CF_DAG_VERTICES, opts)
    }

    /// Create Bullshark certificates column family (round -> certificate)
    fn create_bullshark_cert_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_BULLSHARK_CERT, opts)
    }

    /// Create balance_root_v2 SMT column family.
    ///
    /// Stores Sparse Merkle Tree internal nodes (path_prefix -> 32-byte hash)
    /// plus the persisted root pointer at `\xff__root__`. Reads are random-access
    /// (each proof walks 256 internal nodes); writes happen in batches keyed by
    /// the wallet address path. Lz4 for compression — most values are 32 bytes,
    /// but the tree gets up to 256× the wallet count of internal nodes so
    /// compression helps.
    fn create_balance_smt_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16));
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::balance_smt::CF_BALANCE_SMT, opts)
    }

    /// Create manifest column family (metadata -> value)
    fn create_manifest_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::None); // Small data
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_MANIFEST, opts)
    }

    /// Create transactions column family (tx_id -> transaction)
    fn create_transactions_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(Self::scale_write_buffer(64)); // 64MB write buffer
        opts.set_target_file_size_base(128 * 1024 * 1024); // 128MB target file size
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_TRANSACTIONS, opts)
    }

    /// Create balances column family (wallet_address -> balance)
    /// v0.8.1-beta: Balance consensus storage for mining rewards and transfers
    fn create_balances_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(Self::scale_write_buffer(64)); // v8.6.0: 64MB write buffer for higher throughput (was 32MB)
        opts.set_target_file_size_base(128 * 1024 * 1024); // v8.6.0: 128MB target file size (was 64MB)

        // Optimize for frequent balance updates
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput (was 2)
        opts.set_level_zero_file_num_compaction_trigger(4); // Compact when 4 files accumulate

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_BALANCES, opts)
    }

    /// Create block hash to height column family (block_hash -> height)
    /// v0.8.3-beta: Block hash index for efficient block lookups by hash
    fn create_block_hash_to_height_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4); // Efficient compression
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB write buffer
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB target file size

        // Optimize for read-heavy workload (hash lookups)
        opts.set_max_write_buffer_number(2); // Dual buffering sufficient
        opts.set_level_zero_file_num_compaction_trigger(4);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_BLOCK_HASH_TO_HEIGHT, opts)
    }

    /// Create AI chats column family (chat:* keys -> chat data)
    fn create_ai_chats_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_AI_CHATS, opts)
    }

    /// Create AI credits column family (credits:* -> wallet credits)
    fn create_ai_credits_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_AI_CREDITS, opts)
    }

    /// Create AI transactions column family (aitx:* -> transaction records)
    fn create_ai_transactions_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_AI_TRANSACTIONS, opts)
    }

    /// Create AI treasury column family (treasury:master -> master wallet balance)
    fn create_ai_treasury_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(2)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_AI_TREASURY, opts)
    }

    /// Create AI attachments column family (attachment:* -> metadata) - v0.9.9-beta
    fn create_ai_attachments_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(11));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_AI_ATTACHMENTS, opts)
    }

    /// Create payment proposals column family (proposal:* -> payment proposals)
    fn create_payment_proposals_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(9)); // "proposal:"

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_PAYMENT_PROPOSALS, opts)
    }

    /// Create payment votes column family (vote:* -> validator votes)
    fn create_payment_votes_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_PAYMENT_VOTES, opts)
    }

    /// Create payment locks column family (lock:* -> payment locks)
    fn create_payment_locks_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // v6.1.0: was missing → 64MB default OOM
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(5));

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_PAYMENT_LOCKS, opts)
    }

    /// Create banned peers column family (peer_id -> BanRecord) - v0.9.7-beta
    /// Stores persistent ban list for ZK proof verification failures
    fn create_banned_peers_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (PeerId + timestamp + reason), optimize for reads
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // 8MB
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_BANNED_PEERS, opts)
    }

    /// Create Narwhal payloads column family (digest -> payload)
    fn create_narwhal_payloads_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Large values, optimize for sequential writes
        opts.set_write_buffer_size(Self::scale_write_buffer(256)); // 256MB
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput (was 2)
        opts.set_target_file_size_base(512 * 1024 * 1024); // 512MB

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_NARWHAL_PAYLOADS, opts)
    }

    /// Create sync certificates column family (sync_id -> SyncCertificate) - v0.9.18-beta
    /// Stores AEGIS-QL sync affirmation certificates for TurboSync
    fn create_sync_certificates_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small-medium values (certificates), optimize for reads during sync
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_SYNC_CERTIFICATES, opts)
    }

    /// Create peer trust column family (peer_id -> TrustMetrics) - v0.9.18-beta
    /// Stores AEGIS-QL peer trust metrics for sync reliability
    fn create_peer_trust_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (metrics), optimize for frequent updates
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // 8MB
        opts.set_max_write_buffer_number(2);

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PEER_TRUST, opts)
    }

    /// Create processed updates column family (update_id -> timestamp) - v0.9.98-beta
    /// Stores processed update IDs for P2P durability and idempotency
    /// AI Expert Consensus: Required to prevent duplicate processing of gossipsub messages
    fn create_processed_updates_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Small values (timestamps), optimize for fast lookups
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // 4MB
        opts.set_max_write_buffer_number(2);
        // TTL could be added in future for automatic cleanup of old update IDs

        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new("processed_updates", opts)
    }

    // ========== v1.0.60-beta: Comprehensive State Sync Column Families ==========

    /// State Merkle Trie nodes for cryptographic state root verification
    fn create_state_trie_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Trie nodes are 65 bytes (internal) or 65 bytes (leaf)
        // Optimize for random reads (proof generation)
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        // v6.0.8: Removed optimize_for_point_lookup (created separate 128MB cache, overridden by shared cache anyway)
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::sparse_merkle_trie::CF_STATE_TRIE, opts)
    }

    /// Token balances (account + token -> balance)
    fn create_token_balances_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Keys are 64 bytes (account + token), values are 8 bytes (u64 balance)
        opts.set_write_buffer_size(Self::scale_write_buffer(64)); // 64MB
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput (was 2)
        // v6.0.8: Removed optimize_for_point_lookup (created separate 256MB cache, overridden by shared cache anyway)
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_TOKEN_BALANCES, opts)
    }

    /// Token metadata (token address -> metadata)
    fn create_tokens_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_TOKENS, opts)
    }

    /// DEX liquidity pools (pool_id -> pool state)
    fn create_dex_pools_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_DEX_POOLS, opts)
    }

    /// LP token balances (pool_id + account -> LP balance)
    fn create_lp_balances_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_LP_BALANCES, opts)
    }

    /// Collateral vaults (vault_id -> vault state)
    fn create_vaults_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_VAULTS, opts)
    }

    /// Oracle price feeds (feed_id -> price data)
    fn create_oracle_prices_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // 8MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_ORACLE_PRICES, opts)
    }

    /// Smart contracts (contract_address -> code hash + metadata)
    fn create_contracts_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_CONTRACTS, opts)
    }

    /// Contract storage (contract_address + key -> value)
    fn create_contract_storage_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Largest CF - contract storage can be huge
        opts.set_write_buffer_size(Self::scale_write_buffer(128)); // 128MB
        opts.set_max_write_buffer_number(4); // v8.6.0: quad buffering for write throughput (was 2)
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_CONTRACT_STORAGE, opts)
    }

    /// AI credits (account -> credits balance + stats)
    fn create_ai_credits_v2_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_AI_CREDITS_V2, opts)
    }

    /// Staking positions (staker + validator -> stake info)
    fn create_stakes_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_STAKES, opts)
    }

    /// Account nonces for replay protection (account -> nonce)
    fn create_nonces_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        // v6.0.8: Removed optimize_for_point_lookup (created separate 64MB cache, overridden by shared cache anyway)
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_NONCES, opts)
    }

    // ========== QNO (Quantum Neural Oracle) Column Families ==========

    /// QNO staking positions: wallet_address:stake_id -> StakingPosition JSON
    fn create_qno_stakes_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_QNO_STAKES, opts)
    }

    /// QNO prediction domains: domain_id -> PredictionDomain JSON
    fn create_qno_domains_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // 8MB - small, rarely changes
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_QNO_DOMAINS, opts)
    }

    /// QNO global statistics: "global" -> StakingStats JSON
    fn create_qno_stats_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(4)); // 4MB - single key
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_QNO_STATS, opts)
    }

    // ========== v2.3.9-beta: Swap History Column Family ==========

    /// Swap history for Token Details Modal transaction history
    /// Key format: "swap:{token}:{timestamp}:{tx_id}" -> JSON swap record
    fn create_swap_history_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB - many small records
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(8)); // "swap:{T}"
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_SWAP_HISTORY, opts)
    }

    // ========== v2.7.9-beta: Perpetual Trading Column Families ==========

    fn create_perp_positions_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PERP_POSITIONS, opts)
    }

    fn create_perp_orders_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PERP_ORDERS, opts)
    }

    fn create_perp_trades_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PERP_TRADES, opts)
    }

    fn create_perp_funding_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PERP_FUNDING, opts)
    }

    fn create_perp_liquidations_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PERP_LIQUIDATIONS, opts)
    }

    /// v3.5.8-beta: Wallet transaction index for decentralized history
    fn create_wallet_tx_index_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_WALLET_TX_INDEX, opts)
    }

    /// v3.5.8-beta: Wallet swap index for decentralized DEX history
    fn create_wallet_swap_index_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_WALLET_SWAP_INDEX, opts)
    }

    /// v3.6.0-beta: Consensus-verified price history for tokens
    /// Key format: [token_address:32][inverted_timestamp:8] for reverse chronological order
    fn create_price_history_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32)); // 32MB - many small price snapshots
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are token address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_PRICE_HISTORY, opts)
    }

    /// v3.9.1-beta: Bank messages storage
    /// Key format: msg_id (string), Value: BankMessage JSON
    fn create_bank_messages_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_BANK_MESSAGES, opts)
    }

    /// v3.9.1-beta: Bank message index by wallet
    /// Key format: [wallet:32][inverted_timestamp:8], Value: msg_id
    fn create_bank_msg_index_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        // Optimize for prefix scans (first 32 bytes are wallet address)
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(32));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_BANK_MSG_INDEX, opts)
    }

    /// v3.9.1-beta: User identity records
    /// Key format: wallet_address (string), Value: UserIdentity JSON
    fn create_user_identities_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_USER_IDENTITIES, opts)
    }

    /// v3.9.1-beta: Death certificates for inheritance
    /// Key format: cert_id (string), Value: DeathCertificate JSON
    fn create_death_certificates_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8)); // 8MB - smaller, less frequent
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_DEATH_CERTIFICATES, opts)
    }

    // v7.3.1: Quantum metadata stored separately for lazy loading
    // Key format: "qm:{height}" (string), Value: bincode-serialized QuantumMetadata
    fn create_quantum_metadata_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16)); // 16MB - one per block
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_QUANTUM_METADATA, opts)
    }

    // ========== v7.3.2: Quillon Mail CFs ==========

    fn create_emails_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(32));
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_EMAILS, opts)
    }

    fn create_emails_by_wallet_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16));
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(64));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_EMAILS_BY_WALLET, opts)
    }

    fn create_emails_by_folder_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16));
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(64));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_EMAILS_BY_FOLDER, opts)
    }

    fn create_email_contacts_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8));
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(64));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_EMAIL_CONTACTS, opts)
    }

    fn create_email_outbound_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16));
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(crate::CF_EMAIL_OUTBOUND, opts)
    }

    // ========== v7.3.3: Blockchain Calendar ==========

    fn create_calendar_events_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(16));
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_CALENDAR_EVENTS, opts)
    }

    fn create_calendar_by_date_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8));
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(64));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_CALENDAR_BY_DATE, opts)
    }

    fn create_calendar_scheduled_tx_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8));
        opts.set_max_write_buffer_number(2);
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(64));
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_CALENDAR_SCHEDULED_TX, opts)
    }

    fn create_calendar_community_cf(cache: &rocksdb::Cache) -> ColumnFamilyDescriptor {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(Self::scale_write_buffer(8));
        opts.set_max_write_buffer_number(2);
        Self::apply_shared_block_cache(&mut opts, cache);
        ColumnFamilyDescriptor::new(CF_CALENDAR_COMMUNITY, opts)
    }

    /// Get column family handle (public for transactions - v0.8.1-beta)
    /// v7.1.2: Auto-creates missing CFs instead of failing, preventing batch write failures
    /// when new CFs are added in code but don't yet exist in the database.
    pub fn get_cf(&self, cf_name: &str) -> Result<Arc<rocksdb::BoundColumnFamily>> {
        // Fast path: CF already exists
        if let Some(cf) = self.db.cf_handle(cf_name) {
            return Ok(cf);
        }

        // Slow path: CF missing - auto-create it with default options
        warn!(
            "⚠️  [CF-AUTO-CREATE] Column family '{}' not found in database, creating on-the-fly",
            cf_name
        );

        let mut cf_opts = Options::default();
        cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        cf_opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB default
        cf_opts.set_max_write_buffer_number(2);

        self.db
            .create_cf(cf_name, &cf_opts)
            .with_context(|| format!(
                "Failed to auto-create column family '{}'. DB path: {}",
                cf_name, self.db_path
            ))?;

        info!(
            "✅ [CF-AUTO-CREATE] Column family '{}' created successfully at runtime",
            cf_name
        );

        // Now fetch the handle - it must exist after successful create_cf
        self.db
            .cf_handle(cf_name)
            .ok_or_else(|| anyhow::anyhow!(
                "Column family '{}' still not found after create_cf succeeded. DB path: {}",
                cf_name, self.db_path
            ))
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

    async fn scan_prefix_seek(&self, cf: &str, prefix: &[u8], limit: usize) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf_handle = self.get_cf(cf)?;
        let effective_limit = if limit == 0 { 10_000 } else { limit };
        let mut results = Vec::with_capacity(effective_limit.min(1024));

        // Use raw iterator with seek — does NOT consult bloom filters.
        // This is the same method ldb scan uses internally.
        // Required for CF_BLOCKS which has bloom filters but no prefix extractor,
        // causing prefix_iterator_cf to give false negatives.
        let iter = self.db.iterator_cf(
            &cf_handle,
            rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key, value) = item.context("Iterator error in scan_prefix_seek")?;
            if !key.starts_with(prefix) {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
            if results.len() >= effective_limit {
                break;
            }
        }

        Ok(results)
    }

    async fn get_dag_blocks_forward(
        &self,
        start_height: u64,
        limit: usize,
    ) -> Result<Vec<(u64, Vec<u8>, Vec<u8>)>> {
        // Returns (height, key, value) tuples — caller does deserialization.
        //
        // v10.3.9: String-sort fix. RocksDB sorts qblock:dag: keys lexicographically,
        // NOT numerically. Keys with more digits sort BEFORE fewer digits because
        // digits (0x30-0x39) are less than ':' (0x3A). So 8-digit heights (10M+)
        // appear first in the iterator even when we want 7-digit heights (1M-9M).
        //
        // Old code: exited as soon as `limit` blocks collected → always returned
        // 8-digit blocks (13M+) regardless of start_height.
        //
        // Fix: scan all MAX_SCAN entries without early exit, then sort numerically
        // and return the `limit` blocks with LOWEST heights >= start_height.
        const MAX_SCAN: usize = 100_000;
        const MAX_LIMIT: usize = 2000;
        let limit = limit.min(MAX_LIMIT);

        let cf_handle = self.get_cf(CF_BLOCKS)?;
        let iter = self.db.iterator_cf(
            &cf_handle,
            rocksdb::IteratorMode::From(b"qblock:dag:", rocksdb::Direction::Forward),
        );

        let mut results: Vec<(u64, Vec<u8>, Vec<u8>)> = Vec::with_capacity(limit * 4);
        let mut seen_heights = std::collections::HashSet::with_capacity(limit * 4);
        let mut scanned = 0usize;

        for item in iter {
            let (key, value) = item.context("Iterator error in get_dag_blocks_forward")?;
            scanned += 1;

            if scanned > MAX_SCAN { break; }
            if !key.starts_with(b"qblock:dag:") { break; }

            // Parse height from key bytes without allocation
            let height = match parse_dag_key_height(&key) {
                Some(h) => h,
                None => continue,
            };

            if height < start_height { continue; }
            if seen_heights.contains(&height) { continue; }

            seen_heights.insert(height);
            results.push((height, key.to_vec(), value.to_vec()));
            // No early exit — must scan full MAX_SCAN to collect all digit-length
            // groups before picking the lowest-height ones numerically.
        }

        // Sort numerically and return the limit lowest heights >= start_height.
        results.sort_by_key(|(h, _, _)| *h);
        results.truncate(limit);

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
        // v8.7.2: Run compaction on a blocking thread to avoid starving the tokio runtime.
        // compact_range_cf is synchronous and can take minutes on large DBs — if it runs
        // on a tokio worker thread it blocks all HTTP requests, causing 504 timeouts.
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let cf_names = vec![
                "default",
                "blocks",
                "dag_vertices",
                "bullshark_cert",
                "manifest",
            ];
            for cf_name in cf_names {
                if let Some(cf_handle) = db.cf_handle(cf_name) {
                    debug!("🗜️ Compacting column family: {}", cf_name);
                    db.compact_range_cf(&cf_handle, None::<&[u8]>, None::<&[u8]>);
                }
            }
        }).await.map_err(|e| anyhow::anyhow!("Compaction task failed: {}", e))?;
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
    /// v10.3.15: Re-index all qblock:dag:{N}:{proposer} keys to qblock:height:{N}.
    ///
    /// 545,710 early-history blocks exist only as DAG-format keys. The
    /// get_dag_blocks_forward() scan budget (100K entries) fills with 10M+ blocks
    /// first due to string-sort ordering (8-digit keys starting with '1' precede
    /// 7-digit keys because ':' (0x3A) > '0' (0x30)), so 1M-range blocks are never
    /// returned when a fresh node requests them — keeping contiguous pointer at 0.
    ///
    /// Fix: alias each qblock:dag:{N}:{proposer} to qblock:height:{N} so that
    /// get_qblocks_range() (multi_get fast path) finds them without any iterator.
    ///
    /// Properties: idempotent (skips existing height keys), additive (never deletes),
    /// self-checkpointing (CF_MANIFEST migration flag prevents re-run on restart).
    pub async fn reindex_dag_blocks_to_height_keys(&self) -> Result<u64> {
        // Check migration flag in a scoped block so BoundColumnFamily is dropped
        // before the spawn_blocking await (required for Send bound).
        {
            let cf = self.get_cf(CF_MANIFEST)?;
            if self.db.get_cf(&cf, b"dag_height_reindex_v1").unwrap_or(None).is_some() {
                info!("✅ [REINDEX] DAG→height re-index already complete, skipping");
                return Ok(0);
            }
        } // cf dropped here — nothing non-Send held across the await below

        let db = self.db.clone();

        let written = tokio::task::spawn_blocking(move || -> Result<u64> {
            let cf_blocks = db.cf_handle(CF_BLOCKS)
                .ok_or_else(|| anyhow::anyhow!("CF_BLOCKS not found in DAG reindex"))?;
            let cf_manifest = db.cf_handle(CF_MANIFEST)
                .ok_or_else(|| anyhow::anyhow!("CF_MANIFEST not found in DAG reindex"))?;

            info!("🔄 [REINDEX] DAG→height re-index starting (one-time, may take a few minutes)...");
            let start = std::time::Instant::now();

            let iter = db.iterator_cf(
                &cf_blocks,
                rocksdb::IteratorMode::From(b"qblock:dag:", rocksdb::Direction::Forward),
            );

            let mut scanned = 0u64;
            let mut written = 0u64;
            let mut skipped_exists = 0u64;
            let mut write_batch = WriteBatch::default();
            let mut batch_size = 0usize;

            for item in iter {
                let (key, value) = match item {
                    Ok(kv) => kv,
                    Err(e) => { warn!("⚠️ [REINDEX] Iterator error: {}", e); continue; }
                };

                if !key.starts_with(b"qblock:dag:") { break; }
                scanned += 1;

                let height = match parse_dag_key_height(&key) {
                    Some(h) => h,
                    None => continue,
                };

                let height_key = format!("qblock:height:{}", height);

                // Skip heights already present as height keys
                match db.get_cf(&cf_blocks, height_key.as_bytes()) {
                    Ok(Some(_)) => { skipped_exists += 1; continue; }
                    Ok(None) => {}
                    Err(_) => continue,
                }

                write_batch.put_cf(&cf_blocks, height_key.as_bytes(), &value);
                batch_size += 1;

                if batch_size >= 500 {
                    let mut wo = rocksdb::WriteOptions::default();
                    wo.set_sync(false);
                    wo.disable_wal(false);
                    db.write_opt(std::mem::replace(&mut write_batch, WriteBatch::default()), &wo)
                        .context("write_batch failed in DAG reindex")?;
                    written += batch_size as u64;
                    batch_size = 0;

                    if written % 5000 < 500 {
                        info!("🔄 [REINDEX] scanned={} written={} skipped={} elapsed={:?}",
                              scanned, written, skipped_exists, start.elapsed());
                    }
                }
            }

            // Final partial batch with fsync
            if batch_size > 0 {
                let mut wo = rocksdb::WriteOptions::default();
                wo.set_sync(true);
                wo.disable_wal(false);
                db.write_opt(write_batch, &wo)
                    .context("final write_batch failed in DAG reindex")?;
                written += batch_size as u64;
            }

            info!("✅ [REINDEX] Complete: scanned={} written={} skipped_exists={} elapsed={:?}",
                  scanned, written, skipped_exists, start.elapsed());

            // Persist migration flag (fsync) so this never re-runs
            let mut fo = rocksdb::WriteOptions::default();
            fo.set_sync(true);
            db.put_cf_opt(&cf_manifest, b"dag_height_reindex_v1", b"done", &fo)
                .context("Failed to write DAG reindex migration flag")?;

            Ok(written)
        }).await??;

        Ok(written)
    }

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

    /// v6.1.3: Report RocksDB memory usage for OOM diagnostics
    /// FIX: Query per-CF and sum results. DB-level property_int_value only queries
    /// the default CF which has no data (all data in named CFs → always returned 0).
    pub fn get_memory_usage_mb(&self) -> (f64, f64, f64) {
        let mut total_memtable: u64 = 0;
        let mut total_readers: u64 = 0;

        // Sum memtable and table reader memory across all column families
        let cf_names = [
            "blocks", "dag_vertices", "bullshark_cert", "manifest",
            "transactions", "balances", "block_hash_to_height",
            "cf_token_balances", "cf_tokens", "cf_dex_pools", "cf_contracts",
            "cf_contract_storage", "cf_stakes", "cf_swap_history",
            "cf_price_history", "cf_wallet_tx_index", "cf_wallet_swap_index",
        ];
        for cf_name in &cf_names {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                if let Ok(Some(v)) = self.db.property_int_value_cf(&cf, "rocksdb.cur-size-all-mem-tables") {
                    total_memtable += v;
                }
                if let Ok(Some(v)) = self.db.property_int_value_cf(&cf, "rocksdb.estimate-table-readers-mem") {
                    total_readers += v;
                }
            }
        }

        // Block cache usage is shared across CFs - query once from any CF
        let block_cache = cf_names.iter()
            .find_map(|cf_name| {
                self.db.cf_handle(cf_name).and_then(|cf| {
                    self.db.property_int_value_cf(&cf, "rocksdb.block-cache-usage")
                        .ok().flatten()
                })
            })
            .unwrap_or(0);

        (
            total_memtable as f64 / 1_048_576.0,
            total_readers as f64 / 1_048_576.0,
            block_cache as f64 / 1_048_576.0,
        )
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
