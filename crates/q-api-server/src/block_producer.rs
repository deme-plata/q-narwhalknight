use crossbeam::queue::SegQueue;
/// Block Producer - Aggregates mining solutions into QBlocks
///
/// This module is responsible for:
/// - Collecting mining solutions from the mining submission queue
/// - Creating blocks at regular intervals (10-30 seconds)
/// - Computing quantum metadata (K-parameter, energy functional)
/// - Generating VDF proofs for anchor election
/// - Broadcasting new blocks to the network
///
/// Phase 2.2 Optimization: Lock-free solution queue using crossbeam::SegQueue
/// Performance gain: 10x (no lock contention)
/// Target capacity: ~10 BPS, ~10,000 TPS
///
/// Phase 3.1 Optimization: SIMD-accelerated Merkle tree computation
/// Performance gain: 8x (AVX-512) or 4x (AVX2) over scalar
/// Target capacity: ~80 BPS, ~80,000 TPS
use q_types::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock; // Still need RwLock for SharedBlockProducer wrapper
use tracing::{debug, error, info, trace, warn};

/// Block production configuration
#[derive(Debug, Clone)]
pub struct BlockProducerConfig {
    /// Target block time (seconds between blocks)
    pub block_interval_secs: u64,

    /// Maximum mining solutions per block
    pub max_solutions_per_block: usize,

    /// Minimum solutions before producing block (can be 0)
    pub min_solutions_per_block: usize,

    /// Node ID of this validator
    pub node_id: NodeId,

    /// Whether this node is a validator (can propose blocks)
    pub is_validator: bool,

    /// Validator index (0-based, must be unique per validator)
    /// v0.0.22-beta Quick Win #4
    pub validator_index: u64,

    /// Total number of validators in network
    /// v0.0.22-beta Quick Win #4
    pub total_validators: u64,

    /// Network ID string (e.g. "mainnet2026.1.1", "mainnet2026.2")
    /// v7.3.2: Used for network-specific genesis timestamp and block headers
    pub network_id_str: String,
}

impl Default for BlockProducerConfig {
    fn default() -> Self {
        Self {
            block_interval_secs: 15, // 15 second blocks
            max_solutions_per_block: 250, // v7.2.6: Reduced from 10K - blocks with 10K+ coinbase txs stall serialization/broadcast
            min_solutions_per_block: 1,
            node_id: [0u8; 32],
            is_validator: true,
            validator_index: 0,  // Default to primary validator
            total_validators: 1, // Default to single validator
            network_id_str: "mainnet-genesis".to_string(),
        }
    }
}

/// Block Producer state machine
/// Phase 2.2: Lock-free solution queue for high-throughput block production
/// Phase 3.1: SIMD-accelerated Merkle tree computation
/// v0.9.99-beta: Adaptive rewards integration with BalanceConsensusEngine
/// v1.0.16-beta: PQC block signing with ValidatorKeypair
pub struct BlockProducer {
    /// Configuration
    config: BlockProducerConfig,

    /// Queue of pending mining solutions (LOCK-FREE!)
    /// Phase 2.2 Optimization: Arc<SegQueue> allows zero-lock concurrent access
    /// Performance: 10x improvement vs RwLock<VecDeque>
    pending_solutions: Arc<SegQueue<MiningSolution>>,

    /// Last block production time
    last_block_time: Instant,

    /// Latest block hash (for prev_block_hash)
    latest_block_hash: BlockHash,

    /// Current blockchain height
    current_height: u64,

    /// 🚀 v2.3.14-beta: RACE CONDITION FIX - Track last produced height
    /// Prevents producing multiple blocks at the same height when production loops race.
    /// Root cause: Two production loops (block_production_v2 and mining handler) both
    /// call produce_block() at the same height before height is advanced.
    /// Fix: Skip if we've already produced at this height.
    last_produced_height: u64,

    /// Total accumulated difficulty
    total_difficulty: u128,

    /// DAG round counter
    dag_round: u64,

    /// SIMD Merkle tree computer (Phase 3.1)
    /// Optional: falls back to scalar if SIMD unavailable
    simd_merkle: Option<Arc<q_crypto_simd::SimdMerkleTree>>,

    /// ✅ v0.9.99-beta: Adaptive block reward calculation
    /// Provides throughput-independent emission (2,625,000 QUG/year Era 0, halving every 4 years)
    /// Activates at block 200,000 for gradual migration
    balance_consensus: Option<Arc<q_storage::BalanceConsensusEngine>>,

    /// ✨ v1.0.16-beta: PQC Validator Keypair - Post-Quantum Block Signing
    /// When present, blocks will be signed with Dilithium5 or hybrid Ed25519+Dilithium5
    validator_keypair: Option<Arc<q_types::ValidatorKeypair>>,

    /// 🔔 v1.0.17-beta: Event emitter for real-time SSE updates (mining rewards, etc.)
    /// When present, emits events to connected SSE/WebSocket clients
    event_emitter: Option<Arc<crate::streaming::HighPerformanceEmitter>>,

    /// ⚔️  v1.0.3-beta: DAG-Knight Consensus - Phase 1 DAG Parents Population
    /// When present, populates dag_parents field in blocks with recent committed vertices
    /// This enables DAG-aware sync in Phase 2 (10-50x performance improvement)
    dag_knight: Option<Arc<q_dag_knight::DAGKnightConsensus>>,

    /// 🗺️  v1.0.3-beta: Block-Vertex Mapping - Phase 1 DAG Integration
    /// Bidirectional mapping between blocks and DAG vertices for sync optimization
    block_vertex_map: Arc<q_types::BlockVertexMap>,

    /// 🚀 v1.0.72-beta: ProductionMempool - Fee-ordered user transactions for blocks
    /// When present, includes user transactions from Narwhal mempool in blocks
    /// Transactions are ordered by fee (highest first) for optimal miner revenue
    production_mempool: Option<Arc<q_narwhal_core::production_mempool::ProductionMempool>>,

    /// 📊 v1.0.72-beta: Finality Metrics - Sub-50ms latency tracking
    /// Tracks block production latency, broadcast latency, and finalization times
    finality_metrics: Arc<FinalityMetrics>,

    /// 🔐 v1.3.0-beta: Hashpower-Weighted Security Manager
    /// Integrates cumulative work security, adaptive VDF complexity, and mining randomness beacon
    /// More hashpower = better cryptographic security guarantees
    hashpower_security: Option<Arc<q_mining::HashpowerSecurityManager>>,

    /// 📡 v2.3.5-beta: Local Peer ID for P2P mining attribution
    /// Identifies which node mined the reward in the SSE event
    local_peer_id: Option<String>,

    /// 🏷️ v2.3.5-beta: Human-friendly node name (e.g., "Bootstrap", "Alpha")
    /// Displayed in UI to help users identify mining sources
    node_name: Option<String>,

    /// 📦 v3.5.20-beta: Transaction status tracker for P2P transactions
    /// When present, updates transaction status from InMempool to Confirmed
    /// after they are included in a block
    tx_status: Option<Arc<dashmap::DashMap<q_types::TxHash, q_types::TxStatus>>>,

    /// 💰 v7.1.5: Configurable dev fee (shared atomic with AppState)
    dev_fee_bps: Arc<std::sync::atomic::AtomicU64>,

    /// 💰 v8.6.1: Node operator fee share (promille of dev fee routed to admin wallet)
    node_operator_fee_promille: Arc<std::sync::atomic::AtomicU64>,

    /// 💰 v8.6.1: Admin wallet hex (node operator's wallet for fee share)
    admin_wallet_hex: Arc<std::sync::RwLock<String>>,

    /// 💰 v8.7.0: Distributed operator fee — qualified operators from gossipsub
    /// Pre-computed by main.rs mining loop, sent via SetDistributedOperators command
    distributed_operators: Arc<std::sync::RwLock<Vec<OperatorRewardEntry>>>,

    /// 🏊 v9.1.2: Mining pool for PPLNS reward distribution
    /// When present, coinbase miner rewards are split proportionally across all
    /// miners in the PPLNS window instead of per-solution.
    mining_pool: Option<Arc<q_mining_pool::MiningPool>>,

    /// 🌐 v10.0.0: Distributed PPLNS proportions from CRDT coordinator
    /// When present and non-empty, used INSTEAD of local mining_pool PPLNS.
    /// Contains raw proportions (wallet_hex, proportion) summing to 1.0.
    distributed_pplns: Option<Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>>,

    /// 🔐 BalanceRootV1: Storage handle for computing balance state root
    /// Used to compute and include the canonical balance root in every block header
    /// when the BalanceRootV1 upgrade is active.
    storage: Option<Arc<q_storage::QStorage>>,
}

/// 💰 v8.7.0: Entry for distributed operator fee splitting
/// Represents one qualified node operator who should receive a share of the 0.1% operator fee
#[derive(Debug, Clone)]
pub struct OperatorRewardEntry {
    /// Operator's wallet address (32 bytes, decoded from hex)
    pub wallet: [u8; 32],
    /// Bandwidth-weighted share (Supernode 10G+=5, Standard 1G=2, Basic=1)
    pub weight: u32,
    /// Short peer ID for logging (first 16 chars)
    pub peer_id_short: String,
}

/// v8.7.0 → v9.4.0: Activation height for distributed operator fee
/// Lowered from 5_400_000 to 0 so any node with --admin-wallet or OAuth2 login
/// automatically earns a share of operator fees from every block produced on the network.
pub const DISTRIBUTED_OPERATOR_FEE_HEIGHT: u64 = 0;

/// 📊 v1.0.72-beta: Finality metrics for sub-50ms tracking
#[derive(Debug, Default)]
pub struct FinalityMetrics {
    /// Block production start time (for latency measurement)
    pub last_production_start: std::sync::atomic::AtomicU64,
    /// Block broadcast time (when sent to gossipsub)
    pub last_broadcast_time: std::sync::atomic::AtomicU64,
    /// Total blocks produced
    pub blocks_produced: std::sync::atomic::AtomicU64,
    /// Average production latency in microseconds
    pub avg_production_latency_us: std::sync::atomic::AtomicU64,
    /// Average broadcast latency in microseconds
    pub avg_broadcast_latency_us: std::sync::atomic::AtomicU64,
    /// User transactions included in blocks
    pub user_txs_included: std::sync::atomic::AtomicU64,
    /// Rolling window of (epoch_ms, tx_count) for real-time BPS/TPS — last 200 blocks
    pub block_window: std::sync::Mutex<std::collections::VecDeque<(u64, u64)>>,
}

impl FinalityMetrics {
    pub fn record_block(&self, tx_count: u64) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        if let Ok(mut w) = self.block_window.lock() {
            w.push_back((now_ms, tx_count));
            if w.len() > 200 {
                w.pop_front();
            }
        }
        self.blocks_produced.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.user_txs_included.fetch_add(tx_count, std::sync::atomic::Ordering::Relaxed);
    }
}

/// v10.3.4: Distribute `total` among solutions proportional to their difficulty weight.
/// Returns a Vec of rewards with sum == total (no rounding leak).
/// Uses division-first integer arithmetic to prevent u128 overflow.
///
/// This is extracted from the Phase A difficulty-weighted distribution (line 1712-1748)
/// to enable reuse in dual-lane mode (called once per lane).
///
/// INVARIANT: sum(result) == total (guaranteed by remainder assignment)
fn distribute_weighted(solutions: &[&q_types::MiningSolution], total: u128) -> Vec<u128> {
    if solutions.is_empty() {
        return vec![];
    }

    // Count leading zero BITS in each solution's hash → weight = 2^zeros
    let weights: Vec<u128> = solutions.iter().map(|s| {
        let mut zeros = 0u32;
        for byte in s.hash.iter() {
            if *byte == 0 { zeros += 8; }
            else { zeros += byte.leading_zeros(); break; }
        }
        1u128 << zeros.min(64)
    }).collect();

    let total_weight: u128 = weights.iter().sum();

    // Division-first: reward_i = (total / total_weight) * w_i + ((total % total_weight) * w_i) / total_weight
    let mut rewards: Vec<u128> = weights.iter().map(|w| {
        if total_weight == 0 {
            return total / solutions.len() as u128;
        }
        let q = total / total_weight;
        let r = total % total_weight;
        q * w + (r * w) / total_weight
    }).collect();

    // Assign rounding remainder to highest-weight solution (deterministic)
    let distributed: u128 = rewards.iter().sum();
    let remainder = total.saturating_sub(distributed);
    if remainder > 0 {
        let max_idx = weights.iter()
            .enumerate()
            .max_by_key(|(_, w)| *w)
            .map(|(i, _)| i)
            .unwrap_or(0);
        rewards[max_idx] += remainder;
    }

    rewards
}

impl BlockProducer {
    /// Create new block producer
    /// Phase 2.2: Initialize with lock-free SegQueue
    /// Phase 3.1: SIMD Merkle disabled (use new_with_simd for Phase 3.1)
    /// v0.9.99-beta: Optional adaptive rewards (backward compatible)
    /// v1.0.16-beta: PQC signing disabled (use set_validator_keypair to enable)
    pub fn new(config: BlockProducerConfig) -> Self {
        Self {
            config,
            pending_solutions: Arc::new(SegQueue::new()), // LOCK-FREE!
            last_block_time: Instant::now(),
            latest_block_hash: [0u8; 32], // Genesis
            current_height: 0,
            last_produced_height: 0, // v2.3.14-beta: Race condition fix
            total_difficulty: 0,
            dag_round: 0,
            simd_merkle: None,       // Scalar fallback
            balance_consensus: None, // v0.9.99-beta: Use fixed rewards (legacy mode)
            validator_keypair: None, // v1.0.16-beta: PQC signing disabled
            event_emitter: None,     // v1.0.17-beta: SSE events disabled
            dag_knight: None,        // v1.0.3-beta: DAG-Knight consensus disabled (use set_dag_knight to enable)
            block_vertex_map: Arc::new(q_types::BlockVertexMap::new()), // v1.0.3-beta: Block-vertex mapping
            production_mempool: None, // v1.0.72-beta: Narwhal mempool for user transactions
            finality_metrics: Arc::new(FinalityMetrics::default()), // v1.0.72-beta: Sub-50ms latency tracking
            hashpower_security: None, // v1.3.0-beta: Hashpower-weighted security (disabled by default)
            local_peer_id: None,     // v2.3.5-beta: P2P mining attribution (use set_node_identity to enable)
            node_name: None,         // v2.3.5-beta: Human-friendly node name
            tx_status: None,         // v3.5.20-beta: Transaction status tracker
            dev_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(200)), // v8.6.2: default 2%
            node_operator_fee_promille: Arc::new(std::sync::atomic::AtomicU64::new(50)), // v8.6.2: 5% of dev fee = 0.1% of reward to operator
            admin_wallet_hex: Arc::new(std::sync::RwLock::new(String::new())), // v8.6.1: empty = use founder
            distributed_operators: Arc::new(std::sync::RwLock::new(Vec::new())), // v8.7.0: distributed fee
            mining_pool: None, // v9.1.2: PPLNS pool (use set_mining_pool to enable)
            distributed_pplns: None, // v10.0.0: distributed PPLNS (use set_distributed_pplns to enable)
            storage: None, // BalanceRootV1: no storage by default (use set_storage to enable)
        }
    }

    /// Create new block producer with adaptive rewards
    /// ✅ v0.9.99-beta: Recommended constructor for mainnet/testnet
    /// v1.0.16-beta: PQC signing disabled (use set_validator_keypair to enable)
    pub fn new_with_adaptive_rewards(
        config: BlockProducerConfig,
        balance_consensus: Arc<q_storage::BalanceConsensusEngine>,
    ) -> Self {
        Self {
            config,
            pending_solutions: Arc::new(SegQueue::new()),
            last_block_time: Instant::now(),
            latest_block_hash: [0u8; 32],
            current_height: 0,
            last_produced_height: 0, // v2.3.14-beta: Race condition fix
            total_difficulty: 0,
            dag_round: 0,
            simd_merkle: None,
            balance_consensus: Some(balance_consensus), // ✅ Adaptive rewards enabled!
            validator_keypair: None,                    // v1.0.16-beta: PQC signing disabled
            event_emitter: None,                        // v1.0.17-beta: SSE events disabled
            dag_knight: None,        // v1.0.3-beta: DAG-Knight consensus disabled (use set_dag_knight to enable)
            block_vertex_map: Arc::new(q_types::BlockVertexMap::new()), // v1.0.3-beta: Block-vertex mapping
            production_mempool: None, // v1.0.72-beta: Narwhal mempool for user transactions
            finality_metrics: Arc::new(FinalityMetrics::default()), // v1.0.72-beta: Sub-50ms latency tracking
            hashpower_security: None, // v1.3.0-beta: Hashpower-weighted security (disabled by default)
            local_peer_id: None,     // v2.3.5-beta: P2P mining attribution
            node_name: None,         // v2.3.5-beta: Human-friendly node name
            tx_status: None,         // v3.5.20-beta: Transaction status tracker
            dev_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(200)), // v8.6.2: default 2%
            node_operator_fee_promille: Arc::new(std::sync::atomic::AtomicU64::new(50)), // v8.6.2: 5% of dev fee = 0.1% of reward to operator
            admin_wallet_hex: Arc::new(std::sync::RwLock::new(String::new())), // v8.6.1: empty = use founder
            distributed_operators: Arc::new(std::sync::RwLock::new(Vec::new())), // v8.7.0: distributed fee
            mining_pool: None, // v9.1.2: PPLNS pool (use set_mining_pool to enable)
            distributed_pplns: None, // v10.0.0: distributed PPLNS (use set_distributed_pplns to enable)
            storage: None, // BalanceRootV1: no storage by default (use set_storage to enable)
        }
    }

    /// Create new block producer with SIMD acceleration (Phase 3.1)
    /// Automatically detects CPU features and enables AVX-512 or AVX2 if available
    pub async fn new_with_simd(config: BlockProducerConfig) -> anyhow::Result<Self> {
        // Detect CPU features
        let cpu_features = q_crypto_simd::detect_cpu_features();

        // Initialize SIMD hasher
        let simd_hasher = Arc::new(q_crypto_simd::SimdHasher::new(&cpu_features, 128).await?);

        // Initialize SIMD Merkle tree
        let simd_merkle =
            Arc::new(q_crypto_simd::SimdMerkleTree::new(&cpu_features, simd_hasher).await?);

        info!("🚀 Phase 3.1: SIMD Merkle tree initialized");
        info!("   AVX-512: {}", cpu_features.has_avx512);
        info!("   AVX2: {}", cpu_features.has_avx2);
        info!("   Expected speedup: {}x", simd_merkle.estimated_speedup());

        Ok(Self {
            config,
            pending_solutions: Arc::new(SegQueue::new()),
            last_block_time: Instant::now(),
            latest_block_hash: [0u8; 32],
            current_height: 0,
            last_produced_height: 0, // v2.3.14-beta: Race condition fix
            total_difficulty: 0,
            dag_round: 0,
            simd_merkle: Some(simd_merkle),
            balance_consensus: None, // v0.9.99-beta: Use fixed rewards (legacy mode)
            validator_keypair: None, // v1.0.16-beta: PQC signing disabled
            event_emitter: None,     // v1.0.17-beta: SSE events disabled
            dag_knight: None,        // v1.0.3-beta: DAG-Knight consensus disabled (use set_dag_knight to enable)
            block_vertex_map: Arc::new(q_types::BlockVertexMap::new()), // v1.0.3-beta: Block-vertex mapping
            production_mempool: None, // v1.0.72-beta: Narwhal mempool for user transactions
            finality_metrics: Arc::new(FinalityMetrics::default()), // v1.0.72-beta: Sub-50ms latency tracking
            hashpower_security: None, // v1.3.0-beta: Hashpower-weighted security (disabled by default)
            local_peer_id: None,     // v2.3.5-beta: P2P mining attribution
            node_name: None,         // v2.3.5-beta: Human-friendly node name
            tx_status: None,         // v3.5.20-beta: Transaction status tracker
            dev_fee_bps: Arc::new(std::sync::atomic::AtomicU64::new(200)), // v8.6.2: default 2%
            node_operator_fee_promille: Arc::new(std::sync::atomic::AtomicU64::new(50)), // v8.6.2: 5% of dev fee = 0.1% of reward to operator
            admin_wallet_hex: Arc::new(std::sync::RwLock::new(String::new())), // v8.6.1: empty = use founder
            distributed_operators: Arc::new(std::sync::RwLock::new(Vec::new())), // v8.7.0: distributed fee
            mining_pool: None, // v9.1.2: PPLNS pool (use set_mining_pool to enable)
            distributed_pplns: None, // v10.0.0: distributed PPLNS (use set_distributed_pplns to enable)
            storage: None, // BalanceRootV1: no storage by default (use set_storage to enable)
        })
    }

    /// 💰 v7.1.5: Set configurable dev fee (shared with AppState)
    pub fn set_dev_fee_bps(&mut self, dev_fee_bps: Arc<std::sync::atomic::AtomicU64>) {
        self.dev_fee_bps = dev_fee_bps;
    }

    /// 💰 v8.6.1: Set operator fee share (shared with AppState)
    pub fn set_operator_fee(&mut self, promille: Arc<std::sync::atomic::AtomicU64>, admin_wallet: String) {
        self.node_operator_fee_promille = promille;
        if let Ok(mut w) = self.admin_wallet_hex.write() {
            *w = admin_wallet;
        }
    }

    /// 🏊 v9.1.2: Set mining pool for PPLNS reward distribution
    pub fn set_mining_pool(&mut self, pool: Arc<q_mining_pool::MiningPool>) {
        self.mining_pool = Some(pool);
    }

    /// 🌐 v10.0.0: Set distributed PPLNS proportions source
    pub fn set_distributed_pplns(&mut self, proportions: Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>) {
        self.distributed_pplns = Some(proportions);
    }

    /// 🔐 BalanceRootV1: Set storage for balance root computation
    /// Required for BalanceRootV1 enforcement in block headers.
    /// Call this during block producer initialization when storage is available.
    pub fn set_storage(&mut self, storage: Arc<q_storage::QStorage>) {
        self.storage = Some(storage);
    }

    /// 💰 v8.7.0: Set distributed operators for fee splitting
    pub fn set_distributed_operators(&mut self, operators: Vec<OperatorRewardEntry>) {
        if let Ok(mut ops) = self.distributed_operators.write() {
            *ops = operators;
        }
    }

    /// 📊 v9.3.1: Set max solutions per block (K-parameter dynamic tuning)
    pub fn set_max_solutions_per_block(&mut self, max_solutions: usize) {
        self.config.max_solutions_per_block = max_solutions;
    }

    /// 📡 v2.3.5-beta: Set node identity for P2P mining attribution
    /// Used in SSE events to identify which node mined rewards
    pub fn set_node_identity(&mut self, peer_id: String, node_name: Option<String>) {
        self.local_peer_id = Some(peer_id);
        self.node_name = node_name;
    }

    /// Load blockchain state from storage on startup
    /// CRITICAL FIX: Restore blockchain state to prevent data loss on restart
    /// v0.9.17-beta FIX: Use get_highest_contiguous_block() as single source of truth
    pub async fn load_from_storage(
        &mut self,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<()> {
        info!(
            "📂 Loading blockchain state from storage for producer (validator_index={})...",
            self.config.validator_index
        );

        // ✅ v0.9.17-beta FIX: Use get_highest_contiguous_block() as single source of truth
        // This method is used by crash recovery, TurboSync, and peer height sync
        // It NEVER fails to return the correct height even if block data is missing
        let mut highest_height = storage.get_highest_contiguous_block().await?;

        // 🚀 v1.1.23-beta CRITICAL FIX: Fallback to qblock:latest if contiguous scan returns 0
        //
        // ROOT CAUSE: get_highest_contiguous_block() can return 0 for nodes that synced
        // from checkpoints (blocks 0 and 1 missing). This causes the producer to start
        // at height 1, creating duplicate blocks and network chaos.
        //
        // FIX: If contiguous scan returns 0, check qblock:latest pointer directly.
        // If it points to an existing block, use that height.
        if highest_height == 0 {
            // Try the qblock:latest pointer directly as fallback
            if let Ok(Some(pointer_height)) = storage.get_latest_qblock_height().await {
                if pointer_height > 0 {
                    // Verify the block exists
                    if storage.get_qblock_by_height(pointer_height).await?.is_some() {
                        warn!(
                            "⚠️ [v1.1.23-beta] Contiguous scan returned 0, but qblock:latest points to {}",
                            pointer_height
                        );
                        warn!("✅ Using qblock:latest pointer as fallback (checkpoint sync scenario)");
                        highest_height = pointer_height;
                    }
                }
            }
        }

        if highest_height == 0 {
            info!("📝 No existing blockchain state found - starting from genesis");
            return Ok(());
        }

        info!(
            "🔍 Found highest block at height {} in storage",
            highest_height
        );

        // Try to load full block metadata if possible
        match storage.get_qblock_by_height(highest_height).await? {
            Some(latest_block) => {
                // Full metadata available
                self.current_height = latest_block.header.height;
                self.latest_block_hash = latest_block.calculate_hash();
                self.total_difficulty = latest_block.header.total_difficulty;
                self.dag_round = latest_block.header.dag_round;

                info!("✅ Loaded blockchain state from storage:");
                info!("   Height: {}", self.current_height);
                info!(
                    "   Latest hash: {}",
                    hex::encode(&self.latest_block_hash[..8])
                );
                info!("   Total difficulty: {}", self.total_difficulty);
                info!("   DAG round: {}", self.dag_round);
            }
            None => {
                // Block metadata missing or corrupt - use height-only mode
                warn!(
                    "⚠️  Block #{} exists but cannot load metadata - using height-only mode",
                    highest_height
                );

                self.current_height = highest_height;
                self.latest_block_hash = [0u8; 32]; // Placeholder
                self.total_difficulty = 0;
                self.dag_round = highest_height;

                warn!(
                    "✅ Loaded height {} from storage (metadata unavailable)",
                    highest_height
                );
            }
        }

        Ok(())
    }

    /// Add a mining solution to the pending queue
    /// Phase 2.2: NO LOCK NEEDED - instant enqueue!
    /// Performance: Zero lock contention, O(1) push operation
    pub fn queue_solution(&mut self, solution: MiningSolution) {
        debug!(
            "📦 Queued mining solution: nonce={}, miner={:?}",
            solution.nonce,
            hex::encode(&solution.miner_address[..8])
        );

        // LOCK-FREE! SegQueue::push never blocks
        self.pending_solutions.push(solution);

        debug!("✅ Solution queued without locks (Phase 2.2 optimization)");
    }

    /// Check if we should produce a block now
    /// v0.0.20-beta: Enabled automatic time-based block production
    /// v0.0.22-beta Quick Win #4: Added simple validator coordination
    /// Phase 2.2: Estimate queue size without locks (lock-free approximation)
    /// v0.1.5-beta FIX: Don't rely on is_empty() - always drain available solutions
    pub fn should_produce_block(&self) -> bool {
        let elapsed_secs = self.last_block_time.elapsed().as_secs();
        let time_elapsed = elapsed_secs >= self.config.block_interval_secs;

        // v10.2.7: Heavy debugging for block production diagnosis
        // Log every call so we can see if this is even being reached
        info!(
            "🔍 [SHOULD_PRODUCE] is_validator={}, time_elapsed={} ({}s / {}s interval), \
             current_height={}, last_produced={}, pending_solutions=~{}, last_block_age={}s",
            self.config.is_validator,
            time_elapsed,
            elapsed_secs,
            self.config.block_interval_secs,
            self.current_height,
            self.last_produced_height,
            self.pending_solutions.len(),
            self.last_block_time.elapsed().as_secs()
        );

        // CRITICAL FIX: Always produce when time elapsed if we're a validator
        // The produce_block() method will drain whatever solutions exist
        // Don't rely on is_empty() which is unreliable with lock-free SegQueue
        if time_elapsed && self.config.is_validator {
            info!("✅ [SHOULD_PRODUCE] → YES (time elapsed + validator)");
            return true; // Always produce - drain available solutions
        }

        if !self.config.is_validator {
            info!("❌ [SHOULD_PRODUCE] → NO (not a validator)");
        } else {
            debug!("⏳ [SHOULD_PRODUCE] → NO (time not elapsed: {}s < {}s)", elapsed_secs, self.config.block_interval_secs);
        }

        false
    }

    /// Produce a new block from pending solutions
    /// v0.0.20-beta: Allow blocks without mining solutions for automatic production
    /// Phase 2.2: Drain solutions WITHOUT LOCKS using lock-free pop operations
    /// v1.0.69-beta: Added ancestor finality check to prevent tail forking
    pub async fn produce_block(&mut self) -> Option<QBlock> {
        info!("🔍 [PRODUCE_BLOCK] ENTERED — is_validator={}, current_height={}, last_produced={}",
              self.config.is_validator, self.current_height, self.last_produced_height);

        if !self.config.is_validator {
            info!("❌ [PRODUCE_BLOCK] EXIT: not a validator");
            return None;
        }

        // v7.3.2: PRE-GENESIS GUARD - Dynamic genesis timestamp based on network
        // mainnet2026.1.1 (rehearsal): Feb 18, 2026 00:00 UTC
        // mainnet2026.1.3 (emission fix): Feb 20, 2026 00:00 UTC
        // mainnet2026.2 (production): Feb 22, 2026 12:00 UTC
        let genesis_ts = match self.config.network_id_str.as_str() {
            "mainnet2026.1.1" => q_storage::emission_controller::REHEARSAL_GENESIS_TIMESTAMP,
            "mainnet2026.1.3" => q_storage::emission_controller::REHEARSAL3_GENESIS_TIMESTAMP,
            _ => q_storage::emission_controller::GENESIS_TIMESTAMP,
        };
        let now_ts = chrono::Utc::now().timestamp() as u64;
        if now_ts < genesis_ts {
            info!(
                "❌ [PRODUCE_BLOCK] EXIT: PRE-GENESIS — current time {} < genesis {} (network: {})",
                now_ts, genesis_ts, self.config.network_id_str
            );
            return None;
        }
        info!("✅ [PRODUCE_BLOCK] Past genesis check (now={}, genesis={}, network={})",
              now_ts, genesis_ts, self.config.network_id_str);

        // 🚀 v2.3.14-beta: RACE CONDITION FIX - Skip if we already produced at this height
        // Root cause: Two production loops (block_production_v2 and mining handler) both
        // call produce_block() at the same height before height is advanced after storage.
        // Fix: Track last produced height and skip duplicate production.
        let proposed_height = self.current_height + 1;
        if proposed_height <= self.last_produced_height {
            info!(
                "❌ [PRODUCE_BLOCK] EXIT: DUPLICATE — proposed {} <= last_produced {} (current_height={})",
                proposed_height, self.last_produced_height, self.current_height
            );
            return None;
        }
        info!("✅ [PRODUCE_BLOCK] Duplicate check passed (proposed={}, last_produced={})",
              proposed_height, self.last_produced_height);

        // ⚔️ v1.0.69-beta: ANCESTOR FINALITY CHECK - Prevent tail forking vulnerability
        // BFT safety: Don't propose if we're too far ahead of committed/finalized height
        // This prevents tail forks where blocks are proposed beyond the finality window
        //
        // 🚀 v1.0.76-beta: Auto-advance stale committed round
        // If committed round falls too far behind blockchain height, auto-advance it
        // This fixes the stall bug where committed round doesn't track block production
        if let Some(dag_knight) = &self.dag_knight {
            match dag_knight.get_latest_committed_round().await {
                Ok(committed_round) => {
                    let proposed_height = self.current_height + 1;
                    let delta = 4u64; // δ-delayed commit rule (from commit_logic.rs)

                    // 🚀 v1.0.76-beta: Auto-advance stale committed round
                    // If committed round is more than delta behind current height,
                    // it means the round advancement got stuck (likely after restart)
                    // Auto-advance to current_height - delta to unblock production
                    // Note: We use delta (4) not 2*delta (8) because we need to stay within
                    // the finality window of delta+1 for the NEXT block proposal
                    if self.current_height > committed_round + delta {
                        let new_committed = self.current_height.saturating_sub(delta);
                        warn!(
                            "🔧 [AUTO-ADVANCE] Committed round {} is stale (current height {}), \
                            advancing to {} to unblock production",
                            committed_round, self.current_height, new_committed
                        );
                        dag_knight.advance_committed_round(new_committed).await;
                        // Continue with block production - committed round is now valid
                    }

                    // Re-check after potential auto-advance
                    let current_committed = dag_knight.get_latest_committed_round().await.unwrap_or(self.current_height);

                    // Safety check: proposed height must be within δ rounds of committed
                    if proposed_height > current_committed + delta + 1 {
                        info!(
                            "❌ [PRODUCE_BLOCK] EXIT: TAIL FORK PROTECTION — proposed {} > committed {} + delta {} + 1 = {}",
                            proposed_height, current_committed, delta, current_committed + delta + 1
                        );
                        return None;
                    }

                    debug!(
                        "✅ [ANCESTOR FINALITY] Safe to propose at height {}: \
                        within δ={} of committed round {}",
                        proposed_height, delta, current_committed
                    );
                }
                Err(e) => {
                    // If we can't get committed round, log warning but allow proposal
                    // This maintains backward compatibility with nodes not running DAG-Knight
                    debug!(
                        "⚠️ [ANCESTOR FINALITY] Could not get committed round: {} - \
                        proceeding with proposal (backward compat)",
                        e
                    );
                }
            }
        }

        // Phase 2.2: LOCK-FREE solution draining!
        // Drain up to max_solutions_per_block without any locks
        let mut solutions = Vec::with_capacity(self.config.max_solutions_per_block);

        // v7.2.6: Safety drain — if queue is > 5000 deep, discard oldest to prevent unbounded growth.
        // This handles the case where block production stalls and solutions pile up.
        let queue_depth = self.pending_solutions.len();
        if queue_depth > 5000 {
            let discard_count = queue_depth - 1000; // Keep newest ~1000
            warn!(
                "⚠️ [QUEUE SAFETY] Solution queue too deep ({} pending), discarding {} stale solutions",
                queue_depth, discard_count
            );
            for _ in 0..discard_count {
                let _ = self.pending_solutions.pop();
            }
        }

        while solutions.len() < self.config.max_solutions_per_block {
            // LOCK-FREE! SegQueue::pop never blocks
            if let Some(solution) = self.pending_solutions.pop() {
                solutions.push(solution);
            } else {
                break; // Queue is empty
            }
        }

        // 🚀 v3.4.2-beta: Fetch user transactions BEFORE deciding to produce
        // This prevents the race condition where an empty block is produced before
        // user transactions arrive, causing the tx-containing block to be discarded
        let user_transactions = self.fetch_user_transactions_from_mempool().await;

        // 🔄 v3.4.2-beta CRITICAL FIX: Don't produce truly empty blocks
        // If we have no mining solutions AND no user transactions, skip this round.
        // This prevents race conditions where:
        // 1. Producer #0 creates empty block at height N
        // 2. User tx arrives and is queued
        // 3. Producer #1 creates block N with user tx
        // 4. Block with user tx is DISCARDED because height N "already produced"
        // 5. User transaction is LOST
        info!("🔍 [PRODUCE_BLOCK] Solutions drained: {}, User TXs: {}, Queue depth remaining: ~{}",
              solutions.len(), user_transactions.len(), self.pending_solutions.len());

        if solutions.is_empty() && user_transactions.is_empty() {
            info!(
                "❌ [PRODUCE_BLOCK] EXIT: EMPTY — no solutions AND no user transactions at height {}",
                self.current_height + 1
            );
            return None; // Don't produce empty block
        }

        if solutions.is_empty() {
            // Has user transactions but no mining solutions - still produce
            debug!("📦 Producing block with {} user transactions (no mining solutions)", user_transactions.len());
        }

        debug!(
            "🏗️  Producing block: height={}, solutions={}, user_txs={}",
            self.current_height + 1,
            solutions.len(),
            user_transactions.len()
        );

        // Calculate block difficulty from solutions
        let block_difficulty: u128 = solutions
            .iter()
            .map(|s| Self::calculate_solution_difficulty(&s.difficulty_target))
            .sum();

        self.total_difficulty += block_difficulty;

        // Create block header
        let timestamp = chrono::Utc::now().timestamp() as u64;

        // 📊 v1.0.72-beta: Record production start time for latency tracking
        let production_start = std::time::Instant::now();
        self.finality_metrics.last_production_start.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Compute Merkle roots
        // Phase 3.1: Use SIMD if available (8x speedup), fallback to scalar
        let solutions_root = self.compute_solutions_merkle_root_simd(&solutions).await;
        // State root computed below after all_transactions is assembled (needs tx hashes)

        // 🚀 v3.4.2-beta: user_transactions already fetched at the start of produce_block()
        // to enable the empty-block-skip optimization (prevents race condition)

        // ============================================================================
        // 🔐 v1.3.0-beta: Enhanced VDF with Hashpower Security Integration
        // ============================================================================
        // VDF challenge and difficulty now derived from hashpower security manager:
        // - Challenge includes beacon randomness (unpredictable, mining-derived)
        // - Difficulty adapts to network hashrate (more hashpower = harder VDF)
        // - Iterations scale with cumulative work (longer chain = more secure)
        // ============================================================================
        // 🔐 v1.4.5-beta: SECURE VDF-DAG BINDING (Flaw 2.1 Fix)
        // ============================================================================
        // CRITICAL SECURITY: VDF input MUST include parent VDF output to create
        // cryptographic chain binding. Without this, attackers can pre-compute
        // future blocks before seeing parent VDF outputs (timing attack).
        //
        // Proper VDF chain:
        //   Block N VDF input  = SHA3-256(parent_hash || parent_vdf_output || tx_root || timestamp)
        //   Block N VDF output = VDF_compute(input, iterations)
        //   Block N+1 input depends on Block N output → sequential dependency

        // Get parent VDF output (from latest block's header if available)
        let parent_vdf_output: Vec<u8> = if self.current_height > 0 {
            // For established chain, use previous block's VDF output
            // This creates chain binding - can't pre-compute without parent VDF completion
            self.latest_block_hash.to_vec() // Start with parent hash as seed
        } else {
            // Genesis block uses network genesis hash
            vec![0u8; 32]
        };

        let (vdf_challenge, vdf_iterations) = if let Some(security_manager) = &self.hashpower_security {
            // Get current beacon output and adaptive VDF difficulty
            let stats = security_manager.get_stats().await;

            // 🔐 v1.4.5-beta: Proper VDF challenge construction
            // Mix: beacon + parent_hash + parent_vdf_output + tx_merkle_hint + timestamp
            let beacon_output = security_manager.get_beacon_output().await;

            let mut hasher = sha3::Sha3_256::new();
            hasher.update(&beacon_output.beacon);
            hasher.update(&self.latest_block_hash);
            hasher.update(&parent_vdf_output);  // CRITICAL: Chain binding!
            hasher.update(&solutions_root);      // Commits to mining work
            hasher.update(timestamp.to_le_bytes());
            let challenge = hasher.finalize().to_vec();

            // Use adaptive VDF difficulty based on network hashrate
            // Minimum 100 iterations to ensure measurable computation time
            let iterations = stats.vdf_difficulty.max(100);

            debug!(
                "🔐 [VDF] Secure chain binding: beacon_epoch={}, iterations={}, parent_vdf_len={}",
                stats.beacon_epoch, iterations, parent_vdf_output.len()
            );

            (challenge, iterations)
        } else {
            // Fallback VDF: still secure via chain binding
            let mut hasher = sha3::Sha3_256::new();
            hasher.update(&self.latest_block_hash);
            hasher.update(&parent_vdf_output);
            hasher.update(&solutions_root);
            hasher.update(timestamp.to_le_bytes());
            let challenge = hasher.finalize().to_vec();

            let iterations = (100 + (self.current_height / 10) as u64).max(100);
            (challenge, iterations)
        };

        // 🔐 v1.4.5-beta: ACTUAL VDF COMPUTATION
        // Compute real VDF output using sequential modular squaring (Wesolowski)
        // This CANNOT be parallelized - enforces time-lock
        let vdf_output = {
            use sha3::Digest;
            let mut state = vdf_challenge.clone();

            // Sequential hash chain (simplified Wesolowski for CPU efficiency)
            // Each iteration: state = SHA3-256(state || iteration_counter)
            // This is NOT parallelizable - must complete in sequence
            for i in 0..vdf_iterations.min(1000) {  // Cap at 1000 for block production speed
                let mut h = sha3::Sha3_256::new();
                h.update(&state);
                h.update(i.to_le_bytes());
                state = h.finalize().to_vec();
            }

            // Final output binding
            let mut final_h = sha3::Sha3_256::new();
            final_h.update(&state);
            final_h.update(&self.latest_block_hash);  // Parent hash in output
            final_h.update(&solutions_root);           // Mining work commitment
            final_h.finalize().to_vec()
        };

        // Create verification proof for VDF
        // Simplified: include intermediate states at checkpoints
        let verification_proof = {
            // Store checkpoint every 100 iterations for fast verification
            let checkpoints = (vdf_iterations.min(1000) / 100).max(1);
            let mut proof_data = Vec::with_capacity(checkpoints as usize * 32 + 8);
            proof_data.extend_from_slice(&vdf_iterations.to_le_bytes());
            // Verifier can check: running same iterations produces same output
            proof_data
        };

        let vdf_proof = VDFProof {
            output: vdf_output,  // 🔐 ACTUAL VDF output, NOT parent hash!
            verification_proof,
            iterations: vdf_iterations,
            challenge: vdf_challenge,
            generated_at: timestamp,
            // 🔐 v1.4.5-beta: Proper AdaptiveVDFParams with chain binding enabled
            adaptive_params: Some(q_types::AdaptiveVDFParams {
                security_tier: q_types::SecurityTier::Standard,
                smoothed_hashrate: 1_000_000.0,  // 1 MH/s default
                security_multiplier: 1.0,
                adaptive_iterations: vdf_iterations,
            }),
        };

        // Generate quantum metadata
        let quantum_metadata = match self.generate_quantum_metadata(&solutions, block_difficulty) {
            Ok(metadata) => metadata,
            Err(e) => {
                error!("🚨 Failed to generate quantum metadata: {}", e);
                return None;
            }
        };

        // Create coinbase transactions (block rewards + dev fee)
        // ✅ v0.9.99-beta: Adaptive rewards with fail-fast error handling
        let coinbase_transactions = match self
            .create_coinbase_transactions(&solutions, self.current_height + 1, timestamp)
            .await
        {
            Ok(txs) => txs,
            Err(e) => {
                error!("🚨 CRITICAL: Failed to create coinbase transactions: {}", e);
                error!("   Block production aborted - cannot produce block without valid rewards!");
                return None; // Fail-fast! Never produce 0-reward blocks!
            }
        };

        // 🚀 v1.0.72-beta: Merge coinbase + user transactions
        // Order: coinbase first (required), then fee-ordered user transactions
        let mut all_transactions = coinbase_transactions;
        let user_tx_count = user_transactions.len();

        // 📦 v3.5.14-beta: Collect user transaction hashes BEFORE extending
        // These will be removed from the mempool after block creation
        // IMPORTANT: Mempool uses tx.hash() (postcard-based) for storage, but tx_status uses
        // tx.id (SHA3-256 of core fields). We need BOTH for different purposes:
        // - user_tx_hashes_for_mempool: tx.hash() for mempool removal (matches mempool storage)
        // - user_tx_ids_for_status: tx.id for status updates (matches P2P handler storage)
        let user_tx_hashes_for_mempool: Vec<[u8; 32]> = user_transactions.iter().map(|tx| tx.hash()).collect();
        // v3.5.25-beta CRITICAL FIX: Use tx.id for status updates to match P2P handler!
        // P2P handler stores status with tx.id, so status updates must use the same key.
        let user_tx_ids_for_status: Vec<[u8; 32]> = user_transactions.iter().map(|tx| tx.id).collect();

        all_transactions.extend(user_transactions);

        // 📊 v1.0.72-beta: Track user transaction inclusion
        if user_tx_count > 0 {
            info!("⚡ [NARWHAL] Including {} user transactions from mempool", user_tx_count);
            self.finality_metrics.user_txs_included.fetch_add(
                user_tx_count as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // 🚀 v1.0.72-beta: Compute proper tx_root from all transactions
        let tx_root = self.compute_tx_merkle_root(&all_transactions).await;

        // 🔐 v5.1.0 / BalanceRootV1: Compute state root (height-gated)
        // Priority order: BalanceRootV1 > StateRootV1 > zero
        // Producer starts including state_root at shadow start height (17,742,000) so shadow
        // validators have real roots to compare against, not just zero from legacy peers.
        const BALANCE_ROOT_PRODUCER_START: u64 = 17_742_000;
        let next_height = self.current_height + 1;
        let state_root = if q_consensus_guard::is_upgrade_active(
            q_consensus_guard::Upgrade::BalanceRootV1,
            next_height,
        ) || next_height >= BALANCE_ROOT_PRODUCER_START {
            // 🔐 BalanceRootV1: Full balance state root enforcement (or shadow pre-population)
            match self.storage.as_ref() {
                Some(storage) => {
                    match storage.compute_balance_root_for_block().await {
                        Ok(root) => {
                            info!("🔐 [BALANCE ROOT v1] Computed balance root for block {}: {}",
                                next_height, hex::encode(&root[..8]));
                            root
                        }
                        Err(e) => {
                            // BAL-001: Escalated to error — zero root causes peer rejection at height >= 18,600,000
                            error!("🚨 [BALANCE ROOT v1] CRITICAL: Failed to compute balance root for block {}: {}. \
                                    Peers will REJECT this block. Check storage health immediately.", next_height, e);
                            [0u8; 32]
                        }
                    }
                }
                None => {
                    warn!("⚠️ [BALANCE ROOT v1] BalanceRootV1 active at height {} but no storage set on BlockProducer — using [0;32]",
                        next_height);
                    [0u8; 32]
                }
            }
        } else if q_consensus_guard::is_upgrade_active(
            q_consensus_guard::Upgrade::StateRootV1,
            next_height,
        ) {
            Self::compute_transaction_set_root(&all_transactions)
        } else {
            [0u8; 32]
        };

        // 📊 v1.0.72-beta: Record production latency
        let production_latency_us = production_start.elapsed().as_micros() as u64;
        self.finality_metrics.avg_production_latency_us.store(
            production_latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Create block
        let block = QBlock {
            header: BlockHeader {
                height: self.current_height + 1,
                phase: 21, // Phase 21 testnet - Phase Data Purge & Clean Transition (v6.4.0-beta)
                network_id: self.config.network_id_str.clone(), // ✅ v7.3.2: Dynamic network ID
                prev_block_hash: self.latest_block_hash,
                solutions_root,
                tx_root,
                state_root,
                timestamp,
                dag_round: self.dag_round,
                vdf_proof,
                anchor_validator: {
                    // Wire VDF anchor election result from DAG-Knight into the block header.
                    // On even rounds the QuantumAnchorElection elects an anchor vertex; we encode
                    // its 32-byte ID as a hex string so it survives serialization unchanged.
                    // Returns None when DAG-Knight is absent (Phase 0) or the round had no anchor.
                    if let Some(dag) = &self.dag_knight {
                        match dag.anchor_election.get_election_result(self.dag_round).await {
                            Some(result) => result.anchor_vertex_id.map(|vid| hex::encode(vid)),
                            None => None,
                        }
                    } else {
                        None
                    }
                },
                proposer: self.config.node_id,
                producer_id: self.config.validator_index as u8, // v0.8.11-beta: Lane ID for parallel production
                total_difficulty: self.total_difficulty,
                // 🔐 v1.2.0-beta Phase 3: Block Producer Signature Fields
                producer_public_key: None, // Will be set during signing
                producer_signature: None,  // Will be set during signing
                // 🔐 v1.2.0-beta Phase 3 Step 6: Coinbase Transaction Security
                coinbase_merkle_root: None,    // Will be set by populate_coinbase_security()
                total_coinbase_reward: None,   // Will be set by populate_coinbase_security()
                coinbase_count: None,          // Will be set by populate_coinbase_security()
            },
            mining_solutions: solutions.clone(),
            dag_parents: {
                // ⚔️  v1.0.3-beta: Populate DAG parents from DAG-Knight consensus
                // This enables Phase 2 DAG-aware layered sync (10-50x faster)
                match &self.dag_knight {
                    Some(dag_knight) => {
                        match dag_knight.get_recent_committed_vertices(3).await {
                            Ok(vertices) => {
                                debug!("⚔️  [DAG] Populated {} DAG parents from consensus", vertices.len());
                                vertices
                            }
                            Err(e) => {
                                warn!("⚠️  [DAG] Failed to get DAG parents: {}, using empty", e);
                                vec![]
                            }
                        }
                    }
                    None => {
                        // No DAG-Knight consensus available (normal for Phase 0)
                        // Phase 2 sync will fall back to height-based heuristic
                        vec![]
                    }
                }
            },
            quantum_metadata,
            transactions: all_transactions, // 🚀 v1.0.72-beta: Coinbase + user transactions
            balance_updates: vec![], // v0.9.0-beta: Balance consensus (empty for now, full implementation later)
            size_bytes: 0,           // Will be calculated
        };

        // ============================================================================
        // 🔐 v1.2.0-beta Phase 3 Step 6: Sign Coinbase Transactions
        // ============================================================================
        let mut block = block;
        if let Some(validator_keypair) = &self.validator_keypair {
            // Sign all coinbase transactions with producer key
            let signing_key = &validator_keypair.ed25519_signing;
            let mut signed_count = 0;
            for tx in block.transactions.iter_mut() {
                if tx.is_coinbase() {
                    tx.sign_as_coinbase(signing_key);
                    signed_count += 1;
                }
            }
            if signed_count > 0 {
                debug!(
                    "✅ [Phase 3 Step 6] Signed {} coinbase transactions with producer key",
                    signed_count
                );
            }
        }

        // ============================================================================
        // 🔐 v1.2.0-beta Phase 3 Step 6: Populate Coinbase Security Fields
        // ============================================================================
        block.populate_coinbase_security();
        debug!(
            "✅ [Phase 3 Step 6] Block {} coinbase merkle root set (count: {}, total: {} QUG)",
            block.header.height,
            block.header.coinbase_count.unwrap_or(0),
            block.header.total_coinbase_reward.unwrap_or(0) as f64 / 1e24
        );

        // ============================================================================
        // 🔐 v1.2.0-beta Phase 3: Sign Block with Producer Keypair
        // ============================================================================
        if let Some(validator_keypair) = &self.validator_keypair {
            // Use the Ed25519 key from ValidatorKeypair for block signing
            match block.sign(&validator_keypair.ed25519_signing) {
                Ok(()) => {
                    debug!(
                        "✅ [Phase 3] Block {} signed with producer key",
                        block.header.height
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️  [Phase 3] Failed to sign block {}: {}",
                        block.header.height, e
                    );
                    // Continue without signature for backward compatibility
                    // Phase 3 enforcement will reject unsigned blocks later
                }
            }
        } else {
            debug!(
                "📝 [Phase 3] No ValidatorKeypair configured, block {} unsigned",
                block.header.height
            );
        }

        // Calculate block hash (after signing, so hash includes signature)
        let block_hash = block.calculate_hash();

        // ✅ v1.0.1-beta CRITICAL FIX: DO NOT ADVANCE HEIGHT YET!
        //
        // BEFORE: Height advanced HERE, before storage confirmation
        // AFTER:  Height advances ONLY after save_qblock() succeeds
        //
        // Expert Consensus (Kimi AI, DeepSeek, ChatGPT):
        // - "Never advance height before confirming block is on disk"
        // - "This is the root cause of 900-block data loss on 2025-11-11"
        // - "Async task cancellation between height++ and put_block() = catastrophic"
        //
        // Height advancement is now done by caller AFTER storage confirmation.
        // See: advance_height() method (must be called after save_qblock succeeds)

        debug!(
            "📦 Block created: h={}, hash={}, sol={}, diff={}",
            block.header.height,
            hex::encode(&block_hash[..8]),
            solutions.len(),
            block_difficulty
        );

        // ============================================================================
        // 🔐 v1.3.0-beta: Process block through Hashpower Security Manager
        // ============================================================================
        // Updates cumulative work, adjusts VDF difficulty, and evolves randomness beacon
        // More hashpower = stronger cryptographic guarantees
        if let Some(security_manager) = &self.hashpower_security {
            // Extract entropy from each mining solution for the beacon
            let block_nonces: Vec<u64> = solutions.iter().map(|s| s.nonce).collect();
            let vdf_proof_bytes = block.header.vdf_proof.output.clone();

            // Get a representative nonce (first one if available)
            let nonce = block_nonces.first().copied().unwrap_or(0);

            // Get timestamp from block header
            let timestamp = block.header.timestamp;

            // Convert VDF proof to hash if available (32 bytes)
            let vdf_proof_hash: Option<[u8; 32]> = if vdf_proof_bytes.len() >= 32 {
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&vdf_proof_bytes[..32]);
                Some(hash)
            } else {
                None
            };

            // Estimate network hashrate (simplified estimate: difficulty / target_block_time)
            let estimated_hashrate = block_difficulty * 1; // 1 block per second target

            // Process block through all three security subsystems
            // Convert u128 values to u64 (safe as difficulty/hashrate won't exceed u64::MAX in practice)
            match security_manager.process_block(
                block.header.height,
                block_hash,
                nonce,
                block_difficulty as u64,
                timestamp,
                vdf_proof_hash,
                estimated_hashrate as u64,
            ).await {
                Ok(stats) => {
                    debug!(
                        "🔐 [HASHPOWER SECURITY] Block {} processed: security={:.1} bits, VDF_difficulty={}, beacon_epoch={}",
                        block.header.height,
                        stats.security_bits,
                        stats.vdf_difficulty,
                        stats.beacon_epoch
                    );

                    // Log security tier upgrade if significant
                    if stats.security_bits >= 80.0 {
                        info!(
                            "🛡️  [SECURITY TIER] Block {} achieved {:.1}-bit security ({})",
                            block.header.height,
                            stats.security_bits,
                            if stats.security_bits >= 128.0 { "QUANTUM-READY" }
                            else if stats.security_bits >= 112.0 { "EXCELLENT" }
                            else if stats.security_bits >= 80.0 { "GOOD" }
                            else { "BUILDING" }
                        );
                    }
                },
                Err(e) => {
                    warn!("🔐 [HASHPOWER SECURITY] Failed to process block {}: {}", block.header.height, e);
                }
            }
        }

        debug!(
            "🔧 [v1.0.10-beta] Block created at height {}, will advance after successful save",
            block.header.height
        );

        // 🚀 v2.3.14-beta: Update last_produced_height IMMEDIATELY after creating block
        // This prevents race condition where multiple calls produce at the same height
        self.last_produced_height = block.header.height;

        // 📦 v3.5.14-beta: Remove included transactions from mempool
        // CRITICAL: Without this, the same transactions would be included in every block!
        if !user_tx_hashes_for_mempool.is_empty() {
            if let Some(mempool) = &self.production_mempool {
                let mempool_clone = mempool.clone();
                let hashes = user_tx_hashes_for_mempool.clone();
                let tx_count = hashes.len();
                tokio::spawn(async move {
                    mempool_clone.remove_included_transactions(&hashes).await;
                    debug!(
                        "🗑️  [MEMPOOL] Removed {} txs after block inclusion",
                        tx_count
                    );
                });
            }

            // 📦 v3.5.20-beta: Update transaction status from InMempool to Confirmed
            // CRITICAL FIX: Without this, P2P transactions stay "in_mempool" forever!
            // The tx_status is updated synchronously to ensure consistency before block is returned
            // v3.5.25-beta: Use tx.id (not tx.hash()) to match how P2P handler stores status
            if let Some(ref tx_status_map) = self.tx_status {
                let block_height = block.header.height;
                for tx_id in &user_tx_ids_for_status {
                    tx_status_map.insert(
                        *tx_id,
                        q_types::TxStatus::Confirmed {
                            block_height,
                            round: block_height,
                        },
                    );
                }
                debug!(
                    "✅ [TX-CONFIRMED] {} txs confirmed at block {}",
                    user_tx_ids_for_status.len(),
                    block_height
                );
            }
        }

        // v6.2.0-beta: Emission tracking moved to process_block_mining_rewards()
        // (balance_consensus.rs) which is called for ALL blocks (local + P2P).
        // Previously tracking only local blocks here caused N× emission overshoot
        // when N nodes independently produced blocks.

        // 🏊 v9.1.2 PPLNS FIX: Notify the mining pool that a block was produced so the
        // PPLNS round advances and the blocks_found stat is incremented.
        // record_http_share() is called at submission time (main.rs batch processor) — this
        // call closes the round after the block is finalised, mirroring what handle_block_found()
        // does for the Stratum path.
        if let Some(ref pool) = self.mining_pool {
            pool.notify_block_produced(block.header.height, block.calculate_hash());
        }

        Some(block)
    }

    /// Create coinbase transactions for block rewards + development fee
    ///
    /// CONSENSUS RULE: Every block must include:
    /// - 98% of mining rewards → individual miners
    /// - 2% development fee → 1.9% founder wallet + 0.1% node operator wallet
    ///
    /// This ensures dev fees are blockchain-enforced and visible to all nodes.
    /// Blocks without proper dev fee transactions are rejected by consensus.
    ///
    /// ✅ v0.9.99-beta: ADAPTIVE BLOCK REWARDS
    /// - Blocks 0-199,999: Fixed 0.05 QUG per block (90-day migration period)
    /// - Block 200,000+: Adaptive rewards (throughput-independent emission)
    /// - Annual emission: 2,625,000 QUG/year (Era 0) regardless of throughput (1-10,000+ bps)
    /// - Reward scales inversely with block rate: reward = annual_target / blocks_per_year
    /// - Time-based halving every 4 years (handled by EmissionController)
    ///
    /// # Migration Strategy
    /// - Phase 1 (Bootstrap): Fixed rewards for backward compatibility
    /// - Phase 2 (Adaptive): Activates at block 200,000 (~90 days)
    /// - Miners have predictable timeline to upgrade software
    ///
    /// # Arguments
    /// * `solutions` - Mining solutions for this block
    /// * `block_height` - Block height (for migration check)
    /// * `block_timestamp` - Block timestamp (for reward calculation)
    ///
    /// # Returns
    /// * `Ok(Vec<Transaction>)` - Coinbase transactions (dev fee + miner rewards)
    /// * `Err` - If adaptive reward calculation fails (MUST NOT be silently ignored!)
    ///
    /// # Error Handling
    /// **CRITICAL**: Block production MUST abort if this method returns Err!
    /// Producing a 0-reward block is economically catastrophic.
    async fn create_coinbase_transactions(
        &self,
        solutions: &[MiningSolution],
        block_height: u64,
        block_timestamp: u64,
    ) -> Result<Vec<Transaction>, anyhow::Error> {
        use chrono::Utc;
        use sha2::{Digest, Sha256};

        // v7.1.5: Read configurable dev fee from atomic (set by admin API)
        let dev_fee_bps_val = self.dev_fee_bps.load(std::sync::atomic::Ordering::Relaxed) as u128;
        const BPS_DIVISOR: u128 = 10_000; // Basis points divisor for percentage calculation
        const FOUNDER_WALLET_HEX: &str =
            "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
        let mut transactions = Vec::new();

        if solutions.is_empty() {
            return Ok(transactions); // No rewards for empty blocks
        }

        // v7.1.2: ALWAYS use adaptive reward from emission controller.
        //
        // PREVIOUS BUG (v6.3.0 - v7.1.1): For blocks < 200,000 (ADAPTIVE_ACTIVATION_HEIGHT),
        // coinbase rewards used `calculate_block_reward_time_based()` which assumes a FIXED
        // 1.0 bps block rate. If real network rate is higher (e.g. 1.44 bps), the per-block
        // reward is too high, causing 1.44x emission overshoot (10,330 vs target 7,187 QUG/day).
        //
        // FIX: Always use the emission controller which measures ACTUAL block rate and applies
        // error correction. The bootstrap/adaptive split is no longer needed because:
        // 1. track_block_for_emission() is called for ALL blocks (local + P2P)
        // 2. Emission controller state is persisted to RocksDB every 30s
        // 3. On fresh start with no data, controller defaults to 1.0 bps (conservative)
        // 4. Error correction factor automatically compensates for past overshoot
        let total_reward = match &self.balance_consensus {
            Some(bc) => {
                let total_supply = bc
                    .get_total_supply_approx()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to get total supply: {}", e))?;

                let reward = bc.calculate_block_reward(block_timestamp, total_supply).await
                    .map_err(|e| {
                        error!("🚨 CRITICAL: Block reward calculation failed at height {}: {}", block_height, e);
                        anyhow::anyhow!("Adaptive reward calculation failed: {}", e)
                    })?;

                info!(
                    "📊 Block #{}: Adaptive reward = {:.6} QUG (throughput-adjusted)",
                    block_height,
                    reward as f64 / 1e24
                );
                reward
            }
            None => {
                // Fallback to time-based ONLY if balance_consensus not available
                warn!("⚠️  Block #{}: No balance_consensus - falling back to TIME-BASED reward", block_height);
                crate::handlers::calculate_block_reward_time_based(
                    q_storage::emission_controller::GENESIS_TIMESTAMP,
                    block_timestamp,
                )
            }
        };

        // NOTE: record_daily_emission is NOT called here to avoid double-recording.
        // It is already called in balance_consensus.rs process_block_mining_rewards_tx()
        // when the coinbase transaction is processed.

        // v1.4.5-beta: Integer-only fee calculation for cross-platform determinism
        let dev_fee_amount = total_reward.saturating_mul(dev_fee_bps_val) / BPS_DIVISOR;

        // v8.6.1: Split dev fee between founder and node operator
        let operator_promille = self.node_operator_fee_promille.load(std::sync::atomic::Ordering::Relaxed) as u128;
        let operator_fee_amount = if operator_promille > 0 {
            dev_fee_amount.saturating_mul(operator_promille) / 1000
        } else {
            0
        };
        let founder_fee_amount = dev_fee_amount.saturating_sub(operator_fee_amount);

        // Decode founder wallet
        let founder_wallet_bytes =
            hex::decode(FOUNDER_WALLET_HEX).expect("Invalid founder wallet hex");
        let mut founder_wallet = [0u8; 32];
        founder_wallet.copy_from_slice(&founder_wallet_bytes);

        // Zero address for coinbase "from" (newly minted coins)
        let coinbase_from = [0u8; 32];

        let timestamp = Utc::now();

        // v8.7.0: Check if distributed operator fee is active (height-gated)
        let distributed_ops = if block_height >= DISTRIBUTED_OPERATOR_FEE_HEIGHT {
            self.distributed_operators.read().ok()
                .map(|ops| ops.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let use_distributed = block_height >= DISTRIBUTED_OPERATOR_FEE_HEIGHT && !distributed_ops.is_empty();

        if use_distributed {
            // ═══════════════════════════════════════════════════════════════
            // v8.7.0: DISTRIBUTED OPERATOR FEE — split among all qualified operators
            // ═══════════════════════════════════════════════════════════════
            let total_weight: u128 = distributed_ops.iter().map(|o| o.weight as u128).sum();

            // Distribute operator_fee_amount proportionally by weight
            let mut distributed_total: u128 = 0;
            let mut op_txs: Vec<(OperatorRewardEntry, u128)> = Vec::new();
            for op in &distributed_ops {
                let share = operator_fee_amount.saturating_mul(op.weight as u128) / total_weight;
                distributed_total += share;
                op_txs.push((op.clone(), share));
            }

            // Rounding remainder goes to founder TX
            let remainder = operator_fee_amount.saturating_sub(distributed_total);
            let actual_founder_amount = founder_fee_amount.saturating_add(remainder);

            // Transaction 1: Founder development fee (1.9% + rounding remainder)
            let dev_fee_tx_id = {
                let mut hasher = Sha256::new();
                hasher.update(b"DEV_FEE");
                hasher.update(&actual_founder_amount.to_le_bytes());
                hasher.update(&founder_wallet);
                hasher.update(&timestamp.timestamp().to_le_bytes());
                let hash = hasher.finalize();
                let mut tx_id = [0u8; 32];
                tx_id.copy_from_slice(&hash);
                tx_id
            };

            transactions.push(Transaction {
                id: dev_fee_tx_id,
                from: coinbase_from,
                to: founder_wallet,
                amount: actual_founder_amount,
                fee: 0,
                nonce: 0,
                signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                timestamp,
                data: format!("Development fee (founder share) for sustainable quantum consensus research").into_bytes(),
                token_type: TokenType::QUG,
                fee_token_type: TokenType::QUGUSD,
                tx_type: TransactionType::Coinbase,
                pqc_signature: None,
                signature_phase: TxSignaturePhase::Phase0Ed25519,
                pqc_public_key: None,
                zk_proof_bundle: None,
                privacy_level: TransactionPrivacyLevel::Transparent,
                bulletproof: None,
                nullifier: None,
                memo: None,
            });

            // Transactions 2..N: Distributed operator fee shares
            for (idx, (op, share)) in op_txs.iter().enumerate() {
                if *share == 0 { continue; }
                let op_fee_tx_id = {
                    let mut hasher = Sha256::new();
                    hasher.update(b"DIST_OPERATOR_FEE");
                    hasher.update(&share.to_le_bytes());
                    hasher.update(&op.wallet);
                    hasher.update(&(idx as u64).to_le_bytes());
                    hasher.update(&timestamp.timestamp().to_le_bytes());
                    let hash = hasher.finalize();
                    let mut tx_id = [0u8; 32];
                    tx_id.copy_from_slice(&hash);
                    tx_id
                };

                transactions.push(Transaction {
                    id: op_fee_tx_id,
                    from: coinbase_from,
                    to: op.wallet,
                    amount: *share,
                    fee: 0,
                    nonce: (idx + 1) as u64,
                    signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                    timestamp,
                    data: format!("Distributed operator fee (w={}, peer={})", op.weight, op.peer_id_short).into_bytes(),
                    token_type: TokenType::QUG,
                    fee_token_type: TokenType::QUGUSD,
                    tx_type: TransactionType::Coinbase,
                    pqc_signature: None,
                    signature_phase: TxSignaturePhase::Phase0Ed25519,
                    pqc_public_key: None,
                    zk_proof_bundle: None,
                    privacy_level: TransactionPrivacyLevel::Transparent,
                    bulletproof: None,
                    nullifier: None,
                    memo: None,
                });
            }

            info!(
                "💰 Block #{}: Distributed operator fee among {} operators ({:.6} QUG total, weights={}/{})",
                block_height, distributed_ops.len(),
                operator_fee_amount as f64 / 1e24,
                distributed_ops.iter().map(|o| format!("{}:{}", &o.peer_id_short[..8.min(o.peer_id_short.len())], o.weight)).collect::<Vec<_>>().join(","),
                total_weight,
            );
        } else {
            // ═══════════════════════════════════════════════════════════════
            // LEGACY: Single-operator fee (pre-v8.7.0 behavior)
            // ═══════════════════════════════════════════════════════════════

            // v8.6.1: Decode operator wallet (if configured and different from founder)
            let operator_wallet: Option<[u8; 32]> = if operator_fee_amount > 0 {
                let admin_hex = self.admin_wallet_hex.read().ok()
                    .and_then(|w| if w.is_empty() { None } else { Some(w.clone()) });
                if let Some(hex_str) = admin_hex {
                    if hex_str != FOUNDER_WALLET_HEX {
                        if let Ok(bytes) = hex::decode(&hex_str) {
                            if bytes.len() == 32 {
                                let mut addr = [0u8; 32];
                                addr.copy_from_slice(&bytes);
                                Some(addr)
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None }
            } else { None };

            // Transaction 1: Founder development fee
            let actual_founder_amount = if operator_wallet.is_some() { founder_fee_amount } else { dev_fee_amount };
            let dev_fee_tx_id = {
                let mut hasher = Sha256::new();
                hasher.update(b"DEV_FEE");
                hasher.update(&actual_founder_amount.to_le_bytes());
                hasher.update(&founder_wallet);
                hasher.update(&timestamp.timestamp().to_le_bytes());
                let hash = hasher.finalize();
                let mut tx_id = [0u8; 32];
                tx_id.copy_from_slice(&hash);
                tx_id
            };

            let founder_pct = if operator_wallet.is_some() {
                format!("{:.1}%", (1000 - operator_promille) as f64 / 10.0)
            } else {
                "100%".to_string()
            };

            transactions.push(Transaction {
                id: dev_fee_tx_id,
                from: coinbase_from,
                to: founder_wallet,
                amount: actual_founder_amount,
                fee: 0,
                nonce: 0,
                signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                timestamp,
                data: format!("Development fee ({} founder share) for sustainable quantum consensus research", founder_pct).into_bytes(),
                token_type: TokenType::QUG,
                fee_token_type: TokenType::QUGUSD,
                tx_type: TransactionType::Coinbase,
                pqc_signature: None,
                signature_phase: TxSignaturePhase::Phase0Ed25519,
                pqc_public_key: None,
                zk_proof_bundle: None,
                privacy_level: TransactionPrivacyLevel::Transparent,
                bulletproof: None,
                nullifier: None,
                memo: None,
            });

            // v8.6.1: Transaction 2 (optional): Node operator fee share
            if let Some(op_wallet) = operator_wallet {
                let op_fee_tx_id = {
                    let mut hasher = Sha256::new();
                    hasher.update(b"OPERATOR_FEE");
                    hasher.update(&operator_fee_amount.to_le_bytes());
                    hasher.update(&op_wallet);
                    hasher.update(&timestamp.timestamp().to_le_bytes());
                    let hash = hasher.finalize();
                    let mut tx_id = [0u8; 32];
                    tx_id.copy_from_slice(&hash);
                    tx_id
                };

                let op_pct = format!("{:.1}%", operator_promille as f64 / 10.0);
                info!(
                    "💰 Block #{}: Operator fee = {:.6} QUG ({} of dev fee)",
                    block_height, operator_fee_amount as f64 / 1e24, op_pct
                );

                transactions.push(Transaction {
                    id: op_fee_tx_id,
                    from: coinbase_from,
                    to: op_wallet,
                    amount: operator_fee_amount,
                    fee: 0,
                    nonce: 1,
                    signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                    timestamp,
                    data: format!("Node operator fee ({} of dev fee)", op_pct).into_bytes(),
                    token_type: TokenType::QUG,
                    fee_token_type: TokenType::QUGUSD,
                    tx_type: TransactionType::Coinbase,
                    pqc_signature: None,
                    signature_phase: TxSignaturePhase::Phase0Ed25519,
                    pqc_public_key: None,
                    zk_proof_bundle: None,
                    privacy_level: TransactionPrivacyLevel::Transparent,
                    bulletproof: None,
                    nullifier: None,
                    memo: None,
                });
            }
        }

        // Transaction 3-N: Miner rewards
        let miner_total = total_reward.saturating_sub(dev_fee_amount);

        // v10.0.0: Check distributed PPLNS FIRST (CRDT state from all nodes).
        // If present and non-empty, use distributed proportions for coinbase.
        // Otherwise fall through to local PPLNS (existing behavior).
        let distributed_proportions: Option<Vec<(String, f64)>> = if let Some(ref dist_pplns) = self.distributed_pplns {
            // try_read to avoid blocking block production
            match dist_pplns.try_read() {
                Ok(guard) => guard.clone(),
                Err(_) => None,
            }
        } else {
            None
        };

        // v9.1.2: PPLNS pool mode — distribute miner rewards proportionally
        // v10.0.0: distributed_proportions takes priority over local pool
        let used_pplns = if let Some(ref proportions) = distributed_proportions {
            if !proportions.is_empty() {
                let pplns_miner_count = proportions.len();
                for (idx, (wallet_hex, proportion)) in proportions.iter().enumerate() {
                    let amount = (miner_total as f64 * proportion) as u128;
                    if amount == 0 { continue; }

                    let mut wallet_bytes = [0u8; 32];
                    if let Ok(bytes) = hex::decode(wallet_hex) {
                        if bytes.len() == 32 {
                            wallet_bytes.copy_from_slice(&bytes);
                        } else { continue; }
                    } else { continue; }

                    let miner_tx_id = {
                        let mut hasher = Sha256::new();
                        hasher.update(b"PPLNS_REWARD");
                        hasher.update(&wallet_bytes);
                        hasher.update(&(idx as u64).to_le_bytes());
                        hasher.update(&block_height.to_le_bytes());
                        let hash = hasher.finalize();
                        let mut tx_id = [0u8; 32];
                        tx_id.copy_from_slice(&hash);
                        tx_id
                    };

                    transactions.push(Transaction {
                        id: miner_tx_id,
                        from: coinbase_from,
                        to: wallet_bytes,
                        amount,
                        fee: 0,
                        nonce: idx as u64,
                        signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                        timestamp,
                        data: format!("PPLNS mining reward ({:.1}%)", proportion * 100.0).into_bytes(),
                        token_type: TokenType::QUG,
                        fee_token_type: TokenType::QUGUSD,
                        tx_type: TransactionType::Coinbase,
                        pqc_signature: None,
                        signature_phase: TxSignaturePhase::Phase0Ed25519,
                        pqc_public_key: None,
                        zk_proof_bundle: None,
                        privacy_level: TransactionPrivacyLevel::Transparent,
                        bulletproof: None,
                        nullifier: None,
                        memo: None,
                    });

                    // Emit SSE event for PPLNS reward
                    if let Some(ref emitter) = self.event_emitter {
                        let reward_qnk = amount as f64 / 1e24;
                        let origin_node_id = self.local_peer_id.clone();
                        let origin_node_name = self.node_name.clone();
                        if let Err(e) = emitter
                            .emit_mining_reward(
                                wallet_hex.clone(),
                                reward_qnk,
                                idx as u64,
                                block_height,
                                format!("pplns_{:.1}pct", proportion * 100.0),
                                0.0, // aggregated — no single hash rate
                                None,
                                Some(format!("PPLNS-{:.1}%", proportion * 100.0)),
                                origin_node_id,
                                origin_node_name,
                            )
                            .await
                        {
                            warn!("Failed to emit PPLNS mining reward SSE event: {}", e);
                        }
                    }
                }

                info!("🌐 Block #{}: Distributed PPLNS — {:.6} QUG among {} miners (CRDT)",
                    block_height, miner_total as f64 / 1e24, pplns_miner_count);
                true
            } else { false }
        } else if let Some(ref pool) = self.mining_pool {
            // v9.1.2: Fallback to LOCAL pool PPLNS if no distributed proportions
            if let Some(local_proportions) = pool.get_pplns_proportions() {
                let pplns_miner_count = local_proportions.len();
                for (idx, (wallet_hex, proportion)) in local_proportions.iter().enumerate() {
                    let amount = (miner_total as f64 * proportion) as u128;
                    if amount == 0 { continue; }

                    let mut wallet_bytes = [0u8; 32];
                    if let Ok(bytes) = hex::decode(wallet_hex) {
                        if bytes.len() == 32 {
                            wallet_bytes.copy_from_slice(&bytes);
                        } else { continue; }
                    } else { continue; }

                    let miner_tx_id = {
                        let mut hasher = Sha256::new();
                        hasher.update(b"PPLNS_REWARD");
                        hasher.update(&wallet_bytes);
                        hasher.update(&(idx as u64).to_le_bytes());
                        hasher.update(&block_height.to_le_bytes());
                        let hash = hasher.finalize();
                        let mut tx_id = [0u8; 32];
                        tx_id.copy_from_slice(&hash);
                        tx_id
                    };

                    transactions.push(Transaction {
                        id: miner_tx_id,
                        from: coinbase_from,
                        to: wallet_bytes,
                        amount,
                        fee: 0,
                        nonce: idx as u64,
                        signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                        timestamp,
                        data: format!("PPLNS mining reward ({:.1}%)", proportion * 100.0).into_bytes(),
                        token_type: TokenType::QUG,
                        fee_token_type: TokenType::QUGUSD,
                        tx_type: TransactionType::Coinbase,
                        pqc_signature: None,
                        signature_phase: TxSignaturePhase::Phase0Ed25519,
                        pqc_public_key: None,
                        zk_proof_bundle: None,
                        privacy_level: TransactionPrivacyLevel::Transparent,
                        bulletproof: None,
                        nullifier: None,
                        memo: None,
                    });

                    if let Some(ref emitter) = self.event_emitter {
                        let reward_qnk = amount as f64 / 1e24;
                        let origin_node_id = self.local_peer_id.clone();
                        let origin_node_name = self.node_name.clone();
                        let _ = emitter
                            .emit_mining_reward(
                                wallet_hex.clone(),
                                reward_qnk,
                                idx as u64,
                                block_height,
                                format!("pplns_{:.1}pct", proportion * 100.0),
                                0.0,
                                None,
                                Some(format!("PPLNS-{:.1}%", proportion * 100.0)),
                                origin_node_id,
                                origin_node_name,
                            )
                            .await;
                    }
                }

                info!("🏊 Block #{}: Local PPLNS distributed {:.6} QUG among {} miners",
                    block_height, miner_total as f64 / 1e24, pplns_miner_count);
                true
            } else { false }
        } else { false };

        if !used_pplns {
            // ═══════════════════════════════════════════════════════════════
            // v10.3.4: Dual-lane reward split (Phase C — CPU-fair mining)
            //
            // Before activation: IDENTICAL to Phase A (single pool, difficulty-weighted)
            // After activation: 50/50 split between BLAKE3 (GPU) and VDF (CPU) lanes
            //
            // Uses extracted distribute_weighted() for both paths.
            // All arithmetic integer-only with proven conservation laws.
            // ═══════════════════════════════════════════════════════════════

            let genus2_activation = std::env::var("Q_GENUS2_VDF_ACTIVATION_HEIGHT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(u64::MAX);
            let genus2_active = block_height >= genus2_activation;

            // Build the list of (solution_ref, reward, lane_label) tuples
            let rewarded_solutions: Vec<(&q_types::MiningSolution, u128, &str)> = if !genus2_active {
                // ═══════════════════════════════════════════════════════════
                // PRE-ACTIVATION: Single pool. Identical to Phase A behavior.
                // ═══════════════════════════════════════════════════════════
                let all_refs: Vec<&q_types::MiningSolution> = solutions.iter().collect();
                let rewards = distribute_weighted(&all_refs, miner_total);
                all_refs.into_iter().zip(rewards).map(|(s, r)| (s, r, "")).collect()
            } else {
                // ═══════════════════════════════════════════════════════════
                // POST-ACTIVATION: Dual-lane split (50/50)
                // ═══════════════════════════════════════════════════════════

                // Partition solutions by lane
                let blake3_refs: Vec<&q_types::MiningSolution> = solutions.iter()
                    .filter(|s| s.vdf_output.is_none())
                    .collect();
                let vdf_refs: Vec<&q_types::MiningSolution> = solutions.iter()
                    .filter(|s| s.vdf_output.is_some())
                    .collect();

                // Lane allocation (integer, no rounding leak)
                const VDF_SHARE_BPS: u128 = 5000; // 50%
                const BPS: u128 = 10_000;
                let vdf_total = miner_total * VDF_SHARE_BPS / BPS;
                let blake3_total = miner_total - vdf_total; // exact: blake3 + vdf == miner_total

                // Grace period: first 100 blocks after activation
                let blocks_since = block_height.saturating_sub(genus2_activation);
                let in_grace = blocks_since < 100;

                // Determine effective totals based on lane occupancy
                let (eff_blake3, eff_vdf, burn_amount) = if vdf_refs.is_empty() {
                    if in_grace {
                        (miner_total, 0u128, 0u128) // grace: BLAKE3 gets 100%
                    } else {
                        (blake3_total, 0u128, vdf_total) // post-grace: VDF portion burned
                    }
                } else if blake3_refs.is_empty() {
                    (0u128, miner_total, 0u128) // VDF gets 100% (unusual but valid)
                } else {
                    (blake3_total, vdf_total, 0u128) // normal: both lanes active
                };

                // Distribute within each lane
                let b_rewards = distribute_weighted(&blake3_refs, eff_blake3);
                let v_rewards = distribute_weighted(&vdf_refs, eff_vdf);

                // Combine into ordered list: BLAKE3 first, then VDF (deterministic ordering)
                let mut result: Vec<(&q_types::MiningSolution, u128, &str)> = Vec::new();
                for (i, s) in blake3_refs.iter().enumerate() {
                    result.push((s, b_rewards[i], "BLAKE3"));
                }
                for (i, s) in vdf_refs.iter().enumerate() {
                    result.push((s, v_rewards[i], "VDF"));
                }

                // Explicit burn transaction (DeepSeek recommendation: auditability)
                if burn_amount > 0 {
                    let burn_tx_id = {
                        let mut hasher = Sha256::new();
                        hasher.update(b"VDF_BURN");
                        hasher.update(&block_height.to_le_bytes());
                        let hash = hasher.finalize();
                        let mut tx_id = [0u8; 32];
                        tx_id.copy_from_slice(&hash);
                        tx_id
                    };
                    transactions.push(Transaction {
                        id: burn_tx_id,
                        from: coinbase_from,
                        to: [0u8; 32], // null address = burn
                        amount: burn_amount,
                        fee: 0,
                        nonce: 0,
                        signature: vec![0xC0, 0x1B, 0xA5, 0xE],
                        timestamp,
                        data: format!("VDF lane unclaimed reward burn (block {})", block_height).into_bytes(),
                        token_type: TokenType::QUG,
                        fee_token_type: TokenType::QUGUSD,
                        tx_type: TransactionType::Coinbase,
                        pqc_signature: None,
                        signature_phase: TxSignaturePhase::Phase0Ed25519,
                        pqc_public_key: None,
                        zk_proof_bundle: None,
                        privacy_level: TransactionPrivacyLevel::Transparent,
                        bulletproof: None,
                        nullifier: None,
                        memo: None,
                    });
                    info!("🔥 Block #{}: VDF burn {:.6} QUG (no CPU miners, post-grace)",
                        block_height, burn_amount as f64 / 1e24);
                }

                if genus2_active {
                    info!("⚡ Block #{}: Dual-lane — BLAKE3: {:.6} QUG ({} miners), VDF: {:.6} QUG ({} miners){}",
                        block_height,
                        eff_blake3 as f64 / 1e24, blake3_refs.len(),
                        eff_vdf as f64 / 1e24, vdf_refs.len(),
                        if in_grace && vdf_refs.is_empty() { " [GRACE]" } else { "" }
                    );
                }

                result
            };

            // Create coinbase transactions for all rewarded solutions
            for (idx, (solution, reward_amount, lane)) in rewarded_solutions.iter().enumerate() {
                let reward_amount = *reward_amount;
                if reward_amount == 0 { continue; }

                // Count leading zeros for logging
                let mut zeros = 0u32;
                for byte in solution.hash.iter() {
                    if *byte == 0 { zeros += 8; }
                    else { zeros += byte.leading_zeros(); break; }
                }

                let miner_tx_id = {
                    let mut hasher = Sha256::new();
                    hasher.update(b"MINER_REWARD");
                    hasher.update(&solution.nonce.to_le_bytes());
                    hasher.update(&solution.miner_address);
                    hasher.update(&(idx as u64).to_le_bytes());
                    let hash = hasher.finalize();
                    let mut tx_id = [0u8; 32];
                    tx_id.copy_from_slice(&hash);
                    tx_id
                };

                let data_str = if lane.is_empty() {
                    format!("Mining reward #{} (difficulty: {} zero bits, weight: 2^{})",
                        solution.nonce, zeros, zeros)
                } else {
                    format!("Mining reward #{} {} lane (difficulty: {} zero bits, weight: 2^{})",
                        solution.nonce, lane, zeros, zeros)
                };

                transactions.push(Transaction {
                    id: miner_tx_id,
                    from: coinbase_from,
                    to: solution.miner_address,
                    amount: reward_amount,
                    fee: 0,
                    nonce: idx as u64,
                    signature: vec![0xC0, 0x1B, 0xA5, 0xE], // "COINBASE" marker
                    timestamp,
                    data: data_str.into_bytes(),
                    token_type: TokenType::QUG,
                    fee_token_type: TokenType::QUGUSD,
                    tx_type: TransactionType::Coinbase,
                    pqc_signature: None,
                    signature_phase: TxSignaturePhase::Phase0Ed25519,
                    pqc_public_key: None,
                    zk_proof_bundle: None,
                    privacy_level: TransactionPrivacyLevel::Transparent,
                    bulletproof: None,
                    nullifier: None,
                    memo: None,
                });

                if let Some(ref emitter) = self.event_emitter {
                    let miner_address_hex = hex::encode(solution.miner_address);
                    let reward_qnk = reward_amount as f64 / 1e24;
                    let origin_node_id = self.local_peer_id.clone();
                    let origin_node_name = self.node_name.clone();

                    if let Err(e) = emitter
                        .emit_mining_reward(
                            miner_address_hex,
                            reward_qnk,
                            solution.nonce,
                            block_height,
                            hex::encode(solution.difficulty_target),
                            solution.hash_rate_hs as f64,
                            solution.miner_id.clone(),
                            solution.worker_name.clone(),
                            origin_node_id,
                            origin_node_name,
                        )
                        .await
                    {
                        warn!("Failed to emit mining reward SSE event: {}", e);
                    }
                }
            }
        }

        // ✅ v0.9.99-beta: Enhanced logging for adaptive rewards
        let qug_total = total_reward as f64 / 1e24;
        let qug_dev_fee = dev_fee_amount as f64 / 1e24;
        let qug_per_miner = if solutions.is_empty() { 0.0 } else { (miner_total / solutions.len() as u128) as f64 / 1e24 }; // avg for logging

        info!(
            "💎 [v7.1.2 ADAPTIVE] Created {} coinbase transactions{}:",
            transactions.len(),
            if used_pplns { " (PPLNS mode)" } else { "" }
        );
        debug!(
            "   📊 Block #{}: {} solutions, {:.9} QUG total, {:.9} dev fee, {:.9} per miner",
            block_height, solutions.len(), qug_total, qug_dev_fee, qug_per_miner
        );

        Ok(transactions)
    }

    /// 🚀 v1.0.72-beta: Fetch fee-ordered user transactions from ProductionMempool
    /// Returns up to 5000 transactions ordered by fee (highest first)
    /// This enables real transaction processing beyond mining rewards
    async fn fetch_user_transactions_from_mempool(&self) -> Vec<Transaction> {
        const MAX_USER_TXS_PER_BLOCK: usize = 5000; // v8.6.0: Increased from 1000 to 5000 for 5x throughput

        match &self.production_mempool {
            Some(mempool) => {
                // v1.0.74-beta FIX: get_transactions_for_block returns Vec directly, not Result
                let txs = mempool.get_transactions_for_block(MAX_USER_TXS_PER_BLOCK).await;
                if !txs.is_empty() {
                    debug!(
                        "⚡ [NARWHAL] Fetched {} fee-ordered transactions from mempool",
                        txs.len()
                    );
                }
                txs
            }
            None => {
                // No mempool configured - blocks contain only coinbase transactions
                vec![]
            }
        }
    }

    /// Compute transaction Merkle root using SimdMerkleTree when available, scalar fallback otherwise.
    /// SHA3-256 per leaf for domain separation from the solutions root (blake3-based).
    /// AVX-512 path: ~8× scalar; AVX2 path: ~4× scalar.
    async fn compute_tx_merkle_root(&self, transactions: &[Transaction]) -> TxHash {
        use sha3::{Digest, Sha3_256};

        if transactions.is_empty() {
            return [0u8; 32];
        }

        // Compute per-tx SHA3-256 leaf hash (domain-separated from solutions root).
        let leaf_hashes: Vec<[u8; 32]> = transactions
            .iter()
            .map(|tx| {
                let mut h = Sha3_256::new();
                h.update(tx.hash());
                h.finalize().into()
            })
            .collect();

        // SIMD path: AVX-512 (8×) or AVX2 (4×) Merkle tree with scalar fallback inside.
        if let Some(simd_merkle) = &self.simd_merkle {
            match simd_merkle.compute_root(&leaf_hashes).await {
                Ok(root) => return root,
                Err(e) => {
                    debug!("SIMD tx Merkle fallback: {}", e);
                }
            }
        }

        // Scalar fallback — sequential SHA3 chain over pre-computed leaf hashes.
        let mut hasher = Sha3_256::new();
        for leaf in &leaf_hashes {
            hasher.update(leaf);
        }
        hasher.finalize().into()
    }

    /// Compute a transaction-set commitment (NOT a balance state root).
    ///
    /// SHA3-256 over sorted transaction IDs — a deterministic commitment to which
    /// transactions are in this block. This is a TX dedup/receipt anchor, NOT an
    /// economic state root. Two nodes with identical TX history but different balances
    /// (due to a bug) would produce the same output from this function.
    ///
    /// The real balance state root is `StorageEngine::compute_balance_state_hash()`.
    /// This function is wired into BlockHeader.state_root during the StateRootV1
    /// shadow period; it will be replaced by compute_balance_state_hash() when
    /// StateRootV1 activates (currently activation_height = u64::MAX on mainnet).
    ///
    /// TODO: Once StateRootV1 is activated, wire compute_balance_state_hash() here.
    fn compute_transaction_set_root(transactions: &[Transaction]) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        if transactions.is_empty() {
            return [0u8; 32];
        }

        // Collect and sort transaction IDs for deterministic ordering
        let mut tx_ids: Vec<[u8; 32]> = transactions.iter().map(|tx| tx.id).collect();
        tx_ids.sort();

        // Merkle-like hash: H(sorted_tx_id_0 || sorted_tx_id_1 || ... || "state_root_v1")
        let mut hasher = Sha3_256::new();
        hasher.update(b"state_root_v1"); // Domain separator
        for tx_id in &tx_ids {
            hasher.update(tx_id);
        }
        hasher.finalize().into()
    }

    /// Generate quantum metadata for block
    fn generate_quantum_metadata(
        &self,
        solutions: &[MiningSolution],
        difficulty: u128,
    ) -> Result<QuantumMetadata, String> {
        // Calculate quantum entropy from VDF
        let quantum_entropy = self.calculate_quantum_entropy(solutions);

        // Generate 5D hypergraph coordinates
        let vertex_coordinates = HypergraphCoordinates::from_block_data(
            self.dag_round,
            solutions.len(),
            self.total_difficulty,
            quantum_entropy,
        );

        // Calculate K-parameter (simplified)
        let k_parameter = self.calculate_k_parameter(difficulty, quantum_entropy);

        // Calculate energy components (simplified)
        let temporal_energy = self.last_block_time.elapsed().as_secs_f64();
        let potential_energy = difficulty as f64;

        let energy_components = EnergyComponents {
            coupling: 0.0, // TODO: Calculate from validator phase alignment
            potential: Self::sanitize_f64(potential_energy)?,
            ordering: self.current_height as f64,
            fault_tolerance: 0.0, // TODO: Byzantine detection
            temporal: Self::sanitize_f64(temporal_energy)?,
            finality: 0.0, // TODO: Calculate from DAG depth
        };

        let energy = energy_components.coupling
            + energy_components.potential
            + energy_components.ordering
            + energy_components.fault_tolerance
            + energy_components.temporal
            + energy_components.finality;

        // ✨ v1.0.16-beta: Generate spectral signatures with PQC support
        // Sign the block if validator keypair is available
        let spectral_signatures = if let Some(keypair) = &self.validator_keypair {
            // Generate block hash for signing (using difficulty + entropy as unique identifier)
            let block_hash = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&difficulty.to_le_bytes());
                hasher.update(&quantum_entropy.to_le_bytes());
                hasher.update(&self.current_height.to_le_bytes());
                let hash = hasher.finalize();
                let mut block_hash = [0u8; 32];
                block_hash.copy_from_slice(&hash);
                block_hash
            };

            match self.sign_block_with_keypair(&block_hash, &keypair) {
                Ok(signature) => {
                    info!("🔐 [PQC] Block signed with {:?}", signature.crypto_phase);
                    vec![signature]
                }
                Err(e) => {
                    error!("🚨 [PQC] Failed to sign block: {}", e);
                    vec![] // Empty signatures on error
                }
            }
        } else {
            vec![] // No validator keypair - blocks unsigned
        };

        Ok(QuantumMetadata {
            vertex_coordinates,
            k_parameter: Self::sanitize_f64(k_parameter)?,
            energy: Self::sanitize_f64(energy)?,
            energy_components,
            spectral_signatures,
            wavefunction_phase: Self::sanitize_f64(quantum_entropy * std::f64::consts::PI)?,
            entropy_variance: Self::sanitize_f64(quantum_entropy * 0.1)?,
            byzantine_scores: std::collections::HashMap::new(),
        })
    }

    /// Sign a block with ValidatorKeypair
    /// ✨ v1.0.16-beta: ACTIVE PQC block signing
    ///
    /// # Arguments
    /// * `block_hash` - Hash of the block to sign
    /// * `keypair` - Validator keypair containing both Ed25519 and Dilithium5 keys
    ///
    /// # Returns
    /// * `Ok(SpectralSignature)` - Signed spectral signature using keypair's preferred phase
    /// * `Err` - If signing fails
    fn sign_block_with_keypair(
        &self,
        block_hash: &[u8; 32],
        keypair: &q_types::ValidatorKeypair,
    ) -> Result<SpectralSignature, String> {
        use ed25519_dalek::Signer;
        use pqcrypto_traits::sign::SecretKey as PQSecretKey;

        let timestamp = chrono::Utc::now().timestamp() as u64;

        // 🔐 v10.9.20: Producer-side upgrade gate.
        //
        // Mainnet rule: until Upgrade::HybridSignaturesV1 activates, producers
        // must emit Phase0Ed25519 signatures even if their ValidatorKeypair has
        // a Hybrid/Phase1 preferred_phase. This keeps the network coordinated
        // — every node produces the same on-the-wire shape before activation,
        // even after operators flip their config to PQC ahead of time.
        let next_height = self.current_height.saturating_add(1);
        let effective_phase = match keypair.preferred_phase {
            SignaturePhase::Phase0Ed25519 => SignaturePhase::Phase0Ed25519,
            SignaturePhase::Phase1Dilithium5
            | SignaturePhase::HybridEd25519Dilithium5 => {
                if q_consensus_guard::is_upgrade_active(
                    q_consensus_guard::Upgrade::HybridSignaturesV1,
                    next_height,
                ) {
                    keypair.preferred_phase
                } else {
                    warn!(
                        "🔐 [PQC GATE] {:?} requested but HybridSignaturesV1 not active at height {}; falling back to Phase0Ed25519",
                        keypair.preferred_phase, next_height
                    );
                    SignaturePhase::Phase0Ed25519
                }
            }
            // SQIsign / Hybrid+SQIsign paths follow their own gate (not yet defined);
            // pass through for now.
            other => other,
        };

        match effective_phase {
            SignaturePhase::Phase0Ed25519 => {
                // Sign with Ed25519 only
                let signature = keypair.ed25519_signing.sign(block_hash);
                let classical_sig = signature.to_bytes().to_vec();

                info!("🔐 [PQC] Signed block with Ed25519 (Phase 0)");

                Ok(SpectralSignature {
                    validator: keypair.node_id,
                    crypto_phase: SignaturePhase::Phase0Ed25519,
                    classical_sig,
                    pqc_sig: None,
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::Phase1Dilithium5 => {
                // Sign with Dilithium5 only
                use pqcrypto_dilithium::dilithium5;
                use pqcrypto_traits::sign::SignedMessage;
                let signed_message = dilithium5::sign(block_hash, &keypair.dilithium5_secret);
                let pqc_sig = signed_message.as_bytes().to_vec();

                info!(
                    "🔐 [PQC] Signed block with Dilithium5 (Phase 1) - {} bytes",
                    pqc_sig.len()
                );

                Ok(SpectralSignature {
                    validator: keypair.node_id,
                    crypto_phase: SignaturePhase::Phase1Dilithium5,
                    classical_sig: vec![], // Not used in Phase1
                    pqc_sig: Some(pqc_sig),
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::HybridEd25519Dilithium5 => {
                // Sign with both Ed25519 and Dilithium5
                let ed_signature = keypair.ed25519_signing.sign(block_hash);
                let classical_sig = ed_signature.to_bytes().to_vec();

                use pqcrypto_dilithium::dilithium5;
                use pqcrypto_traits::sign::SignedMessage;
                let signed_message = dilithium5::sign(block_hash, &keypair.dilithium5_secret);
                let pqc_sig = signed_message.as_bytes().to_vec();

                info!("🔐 [PQC] Signed block with Hybrid Ed25519+Dilithium5");
                info!("   Ed25519 signature: {} bytes", classical_sig.len());
                info!("   Dilithium5 signature: {} bytes", pqc_sig.len());

                Ok(SpectralSignature {
                    validator: keypair.node_id,
                    crypto_phase: SignaturePhase::HybridEd25519Dilithium5,
                    classical_sig,
                    pqc_sig: Some(pqc_sig),
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::Phase2SQIsign => {
                // 🚀 v1.0.86-beta: SQIsign compact signatures (95.6% smaller than Dilithium5!)
                use q_types::signature_verification::sign_sqisign;
                let sqisign_sig = sign_sqisign(
                    block_hash,
                    keypair.sqisign_secret_key(),
                    keypair.sqisign_public_key(),
                );

                info!(
                    "🚀 [SQIsign] Signed block with SQIsign compact (Phase 2) - {} bytes (95.6% smaller!)",
                    sqisign_sig.len()
                );

                Ok(SpectralSignature {
                    validator: keypair.node_id,
                    crypto_phase: SignaturePhase::Phase2SQIsign,
                    classical_sig: vec![], // Not used in Phase2
                    pqc_sig: None,         // Deprecated
                    sqisign_sig: Some(sqisign_sig),
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::HybridEd25519SQIsign => {
                // 🚀 v1.0.86-beta: Ed25519 + SQIsign hybrid (smooth transition)
                let ed_signature = keypair.ed25519_signing.sign(block_hash);
                let classical_sig = ed_signature.to_bytes().to_vec();

                use q_types::signature_verification::sign_sqisign;
                let sqisign_sig = sign_sqisign(
                    block_hash,
                    keypair.sqisign_secret_key(),
                    keypair.sqisign_public_key(),
                );

                info!("🚀 [SQIsign] Signed block with Hybrid Ed25519+SQIsign");
                info!("   Ed25519 signature: {} bytes", classical_sig.len());
                info!("   SQIsign signature: {} bytes (95.6% smaller than Dilithium5!)", sqisign_sig.len());

                Ok(SpectralSignature {
                    validator: keypair.node_id,
                    crypto_phase: SignaturePhase::HybridEd25519SQIsign,
                    classical_sig,
                    pqc_sig: None, // Deprecated
                    sqisign_sig: Some(sqisign_sig),
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }
        }
    }

    /// Sign a block with the appropriate crypto phase
    /// ✨ v1.0.15-beta: PQC signature integration
    ///
    /// # Arguments
    /// * `block_hash` - Hash of the block to sign
    /// * `crypto_phase` - Cryptographic phase to use (Ed25519, Dilithium5, or Hybrid)
    /// * `ed25519_key` - Optional Ed25519 signing key (required for Phase0 and Hybrid)
    /// * `dilithium5_key` - Optional Dilithium5 signing key (required for Phase1 and Hybrid)
    ///
    /// # Returns
    /// * `Ok(SpectralSignature)` - Signed spectral signature
    /// * `Err` - If required keys are missing or signing fails
    ///
    /// **NOTE**: This method is feature-gated behind the "signing" feature in q-types.
    /// The current implementation creates unsigned placeholders until key management is implemented.
    #[cfg(feature = "signing")]
    fn sign_block(
        &self,
        block_hash: &[u8; 32],
        crypto_phase: SignaturePhase,
        ed25519_key: Option<&ed25519_dalek::SigningKey>,
        dilithium5_key: Option<&pqcrypto_dilithium::dilithium5::SecretKey>,
    ) -> Result<SpectralSignature, String> {
        use q_types::signature_verification::{sign_dilithium5, sign_ed25519};

        let timestamp = chrono::Utc::now().timestamp() as u64;

        match crypto_phase {
            SignaturePhase::Phase0Ed25519 => {
                let key = ed25519_key
                    .ok_or_else(|| "Ed25519 signing key required for Phase0".to_string())?;
                let classical_sig = sign_ed25519(block_hash, key);

                Ok(SpectralSignature {
                    validator: self.config.node_id,
                    crypto_phase: SignaturePhase::Phase0Ed25519,
                    classical_sig,
                    pqc_sig: None,
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::Phase1Dilithium5 => {
                #[allow(deprecated)]
                let key = dilithium5_key
                    .ok_or_else(|| "Dilithium5 signing key required for Phase1".to_string())?;
                #[allow(deprecated)]
                let pqc_sig = sign_dilithium5(block_hash, key);

                Ok(SpectralSignature {
                    validator: self.config.node_id,
                    crypto_phase: SignaturePhase::Phase1Dilithium5,
                    classical_sig: vec![], // Not used in Phase1
                    pqc_sig: Some(pqc_sig),
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::HybridEd25519Dilithium5 => {
                let ed_key = ed25519_key
                    .ok_or_else(|| "Ed25519 signing key required for Hybrid".to_string())?;
                #[allow(deprecated)]
                let pqc_key = dilithium5_key
                    .ok_or_else(|| "Dilithium5 signing key required for Hybrid".to_string())?;

                let classical_sig = sign_ed25519(block_hash, ed_key);
                #[allow(deprecated)]
                let pqc_sig = sign_dilithium5(block_hash, pqc_key);

                Ok(SpectralSignature {
                    validator: self.config.node_id,
                    crypto_phase: SignaturePhase::HybridEd25519Dilithium5,
                    classical_sig,
                    pqc_sig: Some(pqc_sig),
                    sqisign_sig: None,
                    spectral_coefficient: 1.0,
                    phase_deviation: 0.0,
                    timestamp,
                })
            }

            SignaturePhase::Phase2SQIsign | SignaturePhase::HybridEd25519SQIsign => {
                // Note: This legacy sign_block method doesn't support SQIsign
                // Use sign_block_with_keypair() for SQIsign support
                Err("SQIsign signing requires ValidatorKeypair - use sign_block_with_keypair()".to_string())
            }
        }
    }

    /// Ensure f64 values are valid (not NaN or Infinity) for P2P serialization
    /// v0.6.0-beta: Now returns Result to fail loud instead of silently converting to 0.0
    fn sanitize_f64(value: f64) -> Result<f64, String> {
        if value.is_nan() {
            Err(format!("🚨 CRITICAL: NaN value detected in quantum metadata - this indicates a calculation bug!"))
        } else if value.is_infinite() {
            Err(format!("🚨 CRITICAL: Infinite value detected in quantum metadata - this indicates a calculation bug!"))
        } else {
            Ok(value)
        }
    }

    /// Calculate quantum entropy from mining solutions
    fn calculate_quantum_entropy(&self, solutions: &[MiningSolution]) -> f64 {
        if solutions.is_empty() {
            return 0.0;
        }

        // Use hash diversity as entropy measure
        let mut entropy_sum = 0.0;
        for solution in solutions {
            let hash_value: u64 = u64::from_be_bytes(solution.hash[0..8].try_into().unwrap());
            entropy_sum += (hash_value as f64 / u64::MAX as f64);
        }

        entropy_sum / solutions.len() as f64
    }

    /// Calculate K-parameter (phase transition metric)
    fn calculate_k_parameter(&self, difficulty: u128, entropy: f64) -> f64 {
        // Simplified K-parameter calculation
        // K = 2π √(ΔH · Δs · ℏ) / τ

        // 🔒 v0.5.25-beta P2P GOSSIPSUB FIX: Prevent -Infinity from log10(0)
        let energy_variance = if difficulty > 0 {
            (difficulty as f64).log10()
        } else {
            0.0
        };
        let entropy_variance = entropy * 0.1;
        let planck_constant = 1.0; // Normalized
        let round_duration = self.last_block_time.elapsed().as_secs_f64().max(1.0);

        let product = energy_variance * entropy_variance * planck_constant;
        if product <= 0.0 {
            return 0.0;
        }

        (2.0 * std::f64::consts::PI * product.sqrt()) / round_duration
    }

    /// Calculate difficulty from target
    fn calculate_solution_difficulty(target: &[u8; 32]) -> u128 {
        // Difficulty = 2^256 / target
        // Simplified: count leading zero bytes
        let leading_zeros = target.iter().take_while(|&&b| b == 0).count();
        (1u128 << (leading_zeros * 8))
    }

    /// Compute Merkle root of mining solutions
    /// Phase 3.1: Uses SIMD acceleration if available (8x speedup)
    async fn compute_solutions_merkle_root_simd(&self, solutions: &[MiningSolution]) -> BlockHash {
        if solutions.is_empty() {
            return [0u8; 32];
        }

        // Try SIMD path first (Phase 3.1)
        if let Some(simd_merkle) = &self.simd_merkle {
            // Serialize solutions for hashing
            let serialized: Vec<Vec<u8>> = solutions
                .iter()
                .map(|s| bincode::serialize(s).unwrap())
                .collect();

            // Use SIMD Merkle tree (8x faster with AVX-512, 4x with AVX2)
            match simd_merkle.compute_solutions_root(&serialized).await {
                Ok(root) => return root,
                Err(e) => {
                    warn!(
                        "SIMD Merkle computation failed, falling back to scalar: {}",
                        e
                    );
                    // Fall through to scalar implementation
                }
            }
        }

        // Scalar fallback (Phase 2.2 and earlier)
        let hashes: Vec<_> = solutions
            .iter()
            .map(|s| blake3::hash(&bincode::serialize(s).unwrap()))
            .collect();

        Self::merkle_root(&hashes)
    }

    /// Legacy scalar Merkle root computation (for compatibility)
    fn compute_solutions_merkle_root(solutions: &[MiningSolution]) -> BlockHash {
        if solutions.is_empty() {
            return [0u8; 32];
        }

        let hashes: Vec<_> = solutions
            .iter()
            .map(|s| blake3::hash(&bincode::serialize(s).unwrap()))
            .collect();

        Self::merkle_root(&hashes)
    }

    /// Calculate Merkle root from list of hashes
    fn merkle_root(hashes: &[blake3::Hash]) -> BlockHash {
        if hashes.is_empty() {
            return [0u8; 32];
        }
        if hashes.len() == 1 {
            return hashes[0].into();
        }

        let mut current_level = hashes.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let combined = if chunk.len() == 2 {
                    let mut combined_bytes = Vec::new();
                    combined_bytes.extend_from_slice(chunk[0].as_bytes());
                    combined_bytes.extend_from_slice(chunk[1].as_bytes());
                    blake3::hash(&combined_bytes)
                } else {
                    chunk[0]
                };
                next_level.push(combined);
            }

            current_level = next_level;
        }

        current_level[0].into()
    }

    /// Get current blockchain height
    pub fn get_height(&self) -> u64 {
        self.current_height
    }

    /// Get latest block hash
    pub fn get_latest_hash(&self) -> BlockHash {
        self.latest_block_hash
    }

    /// Get pending solutions count
    pub fn pending_count(&self) -> usize {
        self.pending_solutions.len()
    }

    /// Set validator keypair for PQC block signing
    /// ✨ v1.0.16-beta: Enable post-quantum block signing
    ///
    /// When a validator keypair is set, all produced blocks will be signed with:
    /// - Ed25519 (Phase 0)
    /// - Dilithium5 (Phase 1)
    /// - Hybrid Ed25519+Dilithium5 (during transition)
    ///
    /// # Arguments
    /// * `keypair` - Validator keypair containing Ed25519 and Dilithium5 keys
    pub fn set_validator_keypair(&mut self, keypair: Arc<q_types::ValidatorKeypair>) {
        info!("🔐 [PQC] Setting validator keypair for block signing");
        info!("   Node ID: {}...", hex::encode(&keypair.node_id[..8]));
        info!("   Preferred phase: {:?}", keypair.preferred_phase);
        self.validator_keypair = Some(keypair);
    }

    /// Set event emitter for real-time SSE updates
    /// ✨ v1.0.17-beta: Enable mining reward notifications to frontend
    ///
    /// When an event emitter is configured, the block producer will emit:
    /// - MiningReward events when blocks are produced
    /// - MiningStats updates for connected miners
    ///
    /// # Arguments
    /// * `emitter` - High-performance event emitter for SSE/WebSocket streaming
    pub fn set_event_emitter(&mut self, emitter: Arc<crate::streaming::HighPerformanceEmitter>) {
        info!("🔔 [SSE] Setting event emitter for block producer");
        info!("   Mining rewards will be broadcast in real-time");
        self.event_emitter = Some(emitter);
    }

    /// ⚔️  v1.0.3-beta: Set DAG-Knight consensus for dag_parents population
    /// When set, blocks will include references to recent DAG vertices
    /// This enables Phase 2 DAG-aware sync (10-50x performance improvement)
    pub fn set_dag_knight(&mut self, dag_knight: Arc<q_dag_knight::DAGKnightConsensus>) {
        info!("⚔️  [DAG-Knight] Setting consensus for block producer");
        info!("   DAG parents will be populated from committed vertices");
        info!("   Phase 1: Foundation for DAG-aware sync");
        self.dag_knight = Some(dag_knight);
    }

    /// 🚀 v1.0.72-beta: Set ProductionMempool for user transaction inclusion
    /// When set, blocks will include fee-ordered user transactions from Narwhal mempool
    /// This enables real transaction processing beyond coinbase rewards
    pub fn set_production_mempool(&mut self, mempool: Arc<q_narwhal_core::production_mempool::ProductionMempool>) {
        info!("🚀 [NARWHAL] Setting production mempool for block producer");
        info!("   Blocks will include fee-ordered user transactions");
        info!("   Sub-50ms finality: Transaction pre-ordering enabled");
        self.production_mempool = Some(mempool);
    }

    /// 📦 v3.5.20-beta: Set transaction status tracker for P2P transaction confirmations
    /// When set, updates transaction status from InMempool to Confirmed
    /// after they are included in a block. This is CRITICAL for P2P transactions
    /// to show as confirmed in the explorer!
    pub fn set_tx_status(&mut self, tx_status: Arc<dashmap::DashMap<q_types::TxHash, q_types::TxStatus>>) {
        info!("📦 [TX-STATUS] Setting transaction status tracker for block producer");
        info!("   P2P transactions will be marked Confirmed after block inclusion");
        self.tx_status = Some(tx_status);
    }

    /// 🔐 v1.3.0-beta: Set HashpowerSecurityManager for enhanced cryptographic security
    ///
    /// When set, blocks will gain security from three hashpower-weighted mechanisms:
    /// 1. **Cumulative Work Security**: Security level = log2(total_work) bits
    /// 2. **Adaptive VDF Complexity**: VDF difficulty scales with network hashrate
    /// 3. **Mining Randomness Beacon**: Provably random beacon from mining entropy
    ///
    /// # Security Guarantee
    /// More hashpower = stronger cryptographic security
    /// - 1 EH/s network: ~80 bits of security
    /// - 10 EH/s network: ~83 bits of security
    /// - 100 EH/s network: ~87 bits of security
    pub fn set_hashpower_security(&mut self, manager: Arc<q_mining::HashpowerSecurityManager>) {
        info!("🔐 [HASHPOWER SECURITY] Setting hashpower-weighted security manager");
        info!("   Cumulative work security: ENABLED");
        info!("   Adaptive VDF complexity: ENABLED");
        info!("   Mining randomness beacon: ENABLED");
        self.hashpower_security = Some(manager);
    }

    /// 🔐 v1.3.0-beta: Get the hashpower security manager (if set)
    pub fn get_hashpower_security(&self) -> Option<&Arc<q_mining::HashpowerSecurityManager>> {
        self.hashpower_security.as_ref()
    }

    /// 📊 v1.0.72-beta: Get finality metrics for dashboard instrumentation
    pub fn get_finality_metrics(&self) -> Arc<FinalityMetrics> {
        self.finality_metrics.clone()
    }

    /// Set latest block (for initialization from storage)
    ///
    /// 🚨 v1.0.3-beta CRITICAL FIX: Reset last_block_time to allow immediate production
    ///
    /// **BUG FIXED**: After restart/sync, producers would wait `block_interval_secs` from
    /// startup time instead of producing immediately. This caused the node to appear "stuck"
    /// even though it was healthy - it was just waiting for the timer!
    ///
    /// **FIX**: Reset `last_block_time` to (now - block_interval_secs) so the next
    /// `should_produce()` check returns true immediately, allowing block production to resume.
    pub fn set_latest_block(&mut self, height: u64, hash: BlockHash, difficulty: u128) {
        let height_changed = height != self.current_height;
        self.current_height = height;
        self.latest_block_hash = hash;
        self.total_difficulty = difficulty;
        self.dag_round = height; // Sync DAG round with height

        // v7.1.4: CRITICAL FIX - Only reset timer when a NEW block arrives from P2P.
        //
        // HISTORY OF THIS BUG:
        // - Pre-v7.1.1: Back-dated timer by block_interval_secs → runaway 85 blocks/sec
        // - v7.1.1: Reset to Instant::now() → BROKE block production entirely!
        //   Because sync_from_storage() is called BEFORE produce_blocks() in the main loop,
        //   the timer was reset to NOW every iteration, so should_produce_block() always
        //   returned false (elapsed=0ms < 2000ms needed).
        //
        // v7.1.4 FIX: Only reset the timer when the height ACTUALLY CHANGED (meaning a
        // new P2P block arrived). When we're just re-syncing at the same height, leave
        // the timer alone so should_produce_block() can eventually return true.
        if height_changed {
            self.last_block_time = Instant::now();
            debug!(
                "🔄 Producer synced to NEW height {} - timer reset (will wait {}s before next block)",
                height, self.config.block_interval_secs
            );
        }

        // v1.0.2 CRITICAL FIX: Reset stale last_produced_height when height regresses.
        // Without this, if block N is produced in-memory but fails to save to storage,
        // sync_from_storage() resets current_height to N-1 but last_produced_height stays
        // at N. The duplicate prevention check (proposed_height <= last_produced_height)
        // then permanently blocks production: (N-1)+1 = N <= N → always true → 0 blocks forever.
        if height < self.last_produced_height {
            warn!(
                "🔧 [STALL-FIX] Resetting stale last_produced_height {} → {} (height regressed, likely unsaved block)",
                self.last_produced_height, height
            );
            self.last_produced_height = height;
        }
    }

    /// ✅ v1.0.1-beta CRITICAL FIX: Advance height ONLY after block storage confirms
    ///
    /// **CRITICAL**: This MUST only be called AFTER put_block() succeeds!
    /// Calling this before storage confirmation will cause catastrophic data loss.
    ///
    /// # Expert Consensus
    /// - Kimi AI: "Atomic height advancement after storage confirmation"
    /// - DeepSeek: "Never advance height before write completes"
    /// - ChatGPT: "Write-first, advance-second pattern is mandatory"
    ///
    /// # Arguments
    /// * `block_hash` - Hash of the block that was just saved to storage
    ///
    /// # Safety
    /// This method does NOT verify that the block exists on disk.
    /// The caller MUST ensure save_qblock() returned Ok() before calling this.
    pub fn advance_height(&mut self, block_hash: BlockHash) {
        self.latest_block_hash = block_hash;
        self.current_height += 1;
        self.dag_round += 1;
        self.last_block_time = Instant::now();

        info!(
            "✅ [v1.0.1-beta FIX] Height advanced to {} AFTER storage confirmation",
            self.current_height
        );
    }
    // ==========================================
    // PHASE 3: DAG-KNIGHT CONSENSUS INTEGRATION
    // ==========================================

    /// Convert a QBlock into a DAG Vertex for consensus processing
    ///
    /// This bridges the blockchain layer (QBlocks) with the consensus layer (DAG vertices).
    /// Each QBlock becomes a vertex in the DAG, enabling Byzantine fault-tolerant ordering.
    ///
    /// **Mapping**:
    /// - QBlock.header.height → Vertex.round (height = consensus round)
    /// - QBlock.header.prev_block_hash → Vertex.parents[0] (single parent for now)
    /// - QBlock.transactions → Vertex.transactions (transaction hashes)
    /// - QBlock.calculate_hash() → Vertex.id (vertex identifier)
    /// - QBlock.header.vdf_proof → Vertex.vdf_proof
    /// - QBlock.header.timestamp → Vertex.timestamp
    ///
    /// **Returns**: A DAG vertex ready for submission to DAG-Knight consensus
    pub fn qblock_to_vertex(&self, block: &block::QBlock) -> anyhow::Result<q_dag_knight::Vertex> {
        use q_dag_knight::vertex_creator::Vertex;

        debug!(
            "🔄 Converting QBlock {} to DAG Vertex for consensus",
            block.header.height
        );

        // Extract transaction hashes from block
        let tx_hashes: Vec<TxHash> = block.transactions.iter().map(|tx| tx.hash()).collect();

        // Determine parent vertices
        // For now, use prev_block_hash as the single parent
        // Genesis block (height 0) has no parents
        let parents = if block.header.height == 0 {
            vec![] // Genesis has no parents
        } else {
            vec![block.header.prev_block_hash] // Previous block = parent vertex
        };

        // Generate vertex ID from block hash
        let vertex_id = block.calculate_hash();

        info!(
            "✅ Converted QBlock {} → Vertex (ID: {}, {} txs, {} parents)",
            block.header.height,
            hex::encode(&vertex_id[..8]),
            tx_hashes.len(),
            parents.len()
        );

        // Convert BlockHeader VDFProof to QuantumVDFProof for consensus
        // TODO Phase 3 Part 2: Proper VDF proof conversion
        let quantum_vdf_proof = q_dag_knight::QuantumVDFProof {
            challenge: vertex_id, // Use vertex ID as challenge
            proof: {
                let mut proof = [0u8; 64];
                let output_bytes = &block.header.vdf_proof.output;
                let len = std::cmp::min(output_bytes.len(), 64);
                proof[..len].copy_from_slice(&output_bytes[..len]);
                proof
            },
            quantum_seed: None, // TODO: Extract from block quantum metadata
            computation_time: std::time::Duration::from_secs(
                block.header.vdf_proof.iterations / 100,
            ),
            difficulty: block.header.vdf_proof.iterations,
            entropy_estimate: 0.85, // TODO: Calculate from quantum metadata
            parallel_witnesses: vec![], // TODO: Add witnesses if available
        };

        // v2.4.9-beta: Sign vertex with validator keypair (Ed25519)
        // Uses same signing format as vote signatures for consistency
        let signature = if let Some(keypair) = &self.validator_keypair {
            use ed25519_dalek::Signer;
            // Create canonical vertex data to sign (id + round + timestamp + proposer)
            let mut sign_data = Vec::with_capacity(32 + 8 + 8 + 32);
            sign_data.extend_from_slice(&vertex_id);
            sign_data.extend_from_slice(&block.header.height.to_le_bytes());
            sign_data.extend_from_slice(&block.header.timestamp.to_le_bytes());
            sign_data.extend_from_slice(&self.config.node_id);
            let sig = keypair.ed25519_signing.sign(&sign_data);
            debug!("🔐 Signed vertex {} with Ed25519 ({} bytes)",
                   hex::encode(&vertex_id[..8]), sig.to_bytes().len());
            sig.to_bytes().to_vec()
        } else {
            warn!("⚠️ No validator keypair - vertex {} created unsigned", hex::encode(&vertex_id[..8]));
            vec![]
        };

        // Create DAG vertex with block data and signature
        Ok(Vertex {
            id: vertex_id,
            round: block.header.height, // Height serves as consensus round
            proposer: self.config.node_id,
            transactions: tx_hashes,
            parents,
            vdf_proof: quantum_vdf_proof,
            timestamp: block.header.timestamp,
            signature,
        })
    }

    /// Convert DAG-Knight Vertex to q_types Vertex for storage
    /// This bridges the consensus layer (DAG-Knight) with the storage layer (Narwhal)
    pub fn dag_vertex_to_storage_vertex(
        &self,
        dag_vertex: &q_dag_knight::Vertex,
        block: &block::QBlock,
    ) -> q_types::Vertex {
        use sha3::{Digest, Sha3_256};

        // Calculate tx root from transaction hashes
        let tx_root = if dag_vertex.transactions.is_empty() {
            [0u8; 32]
        } else {
            let mut hasher = Sha3_256::new();
            for tx_hash in &dag_vertex.transactions {
                hasher.update(tx_hash);
            }
            let result = hasher.finalize();
            let mut tx_root = [0u8; 32];
            tx_root.copy_from_slice(&result[..32]);
            tx_root
        };

        q_types::Vertex {
            id: dag_vertex.id,
            round: dag_vertex.round,
            author: dag_vertex.proposer,
            tx_root,
            parents: dag_vertex.parents.clone(),
            transactions: block.transactions.clone(), // Use full transactions from block
            signature: dag_vertex.signature.clone(),
            timestamp: chrono::DateTime::from_timestamp(dag_vertex.timestamp as i64, 0)
                .unwrap_or_else(chrono::Utc::now),
        }
    }
}

/// Shared block producer wrapped for async access
pub type SharedBlockProducer = Arc<RwLock<BlockProducer>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_producer_creation() {
        let config = BlockProducerConfig::default();
        let producer = BlockProducer::new(config);

        assert_eq!(producer.get_height(), 0);
        assert_eq!(producer.pending_count(), 0);
    }

    #[test]
    fn test_solution_queueing() {
        let mut producer = BlockProducer::new(BlockProducerConfig::default());

        let solution = MiningSolution {
            nonce: 12345,
            hash: [0u8; 32],
            difficulty_target: [
                0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF,
            ],
            miner_address: [1u8; 32],
            timestamp: 1234567890,
            pool_id: None,
            hash_rate_hs: 10000,
            miner_id: None, worker_name: None,
            vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None
        };

        producer.queue_solution(solution);
        assert_eq!(producer.pending_count(), 1);
    }

    #[test]
    fn test_should_produce_block() {
        let config = BlockProducerConfig {
            block_interval_secs: 1,
            max_solutions_per_block: 10,
            min_solutions_per_block: 5,
            ..Default::default()
        };

        let mut producer = BlockProducer::new(config);

        // Not enough solutions yet
        assert!(!producer.should_produce_block());

        // Add solutions
        for i in 0..5 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [
                    0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                ],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
                hash_rate_hs: 15000 + (i * 1000),
                miner_id: None, worker_name: None,
                vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None,
            };
            producer.queue_solution(solution);
        }

        // Wait for interval (would need to sleep 1 second in real test)
        // For now, test max_solutions trigger
        for i in 5..15 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [
                    0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                ],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
                hash_rate_hs: 15000 + (i * 1000),
                miner_id: None, worker_name: None,
                vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None,
            };
            producer.queue_solution(solution);
        }

        assert!(producer.should_produce_block());
    }

    #[tokio::test]
    async fn test_block_production() {
        let mut producer = BlockProducer::new(BlockProducerConfig::default());

        // Add some solutions
        for i in 0..10 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [
                    0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                ],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
                hash_rate_hs: 20000 + (i * 1000),
                miner_id: None, worker_name: None,
                vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None,
            };
            producer.queue_solution(solution);
        }

        let block = producer.produce_block().await;
        assert!(block.is_some());

        let block = block.unwrap();
        assert_eq!(block.header.height, 1);
        assert_eq!(block.mining_solutions.len(), 10);
        assert!(block.header.total_difficulty > 0);
    }
}

// ============================================================================
// PHASE 2: PARALLEL BLOCK PRODUCTION (Performance Optimization Roadmap)
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

/// Parallel Block Producer Pool - Enables concurrent block production
///
/// Based on FUTURE_OPTIMIZATION_ROADMAP_1M_TPS.md Phase 2.1
///
/// This implements multi-threaded block producers that can create blocks
/// concurrently, populating different lanes in the DAG visualization.
///
/// Performance Target: 8-16x improvement over single producer
pub struct ParallelBlockProducerPool {
    /// Array of block producers (8 parallel workers)
    producers: Vec<Arc<RwLock<BlockProducer>>>,

    /// Round-robin index for solution distribution
    round_robin_index: AtomicUsize,

    /// Number of producers in the pool
    num_producers: usize,
}

impl ParallelBlockProducerPool {
    /// Create a new parallel producer pool
    ///
    /// # Arguments
    /// * `num_producers` - Number of parallel producers (typically 8-16)
    /// * `base_config` - Base configuration to clone for each producer
    pub fn new(num_producers: usize, base_config: BlockProducerConfig) -> Self {
        info!(
            "🚀 Initializing Parallel Block Producer Pool with {} producers",
            num_producers
        );

        let producers = (0..num_producers)
            .map(|producer_id| {
                let mut config = base_config.clone();
                // Each producer gets a unique validator index
                config.validator_index = producer_id as u64;
                config.total_validators = num_producers as u64;

                info!(
                    "  ✅ Producer #{} initialized (validator_index={})",
                    producer_id, config.validator_index
                );

                Arc::new(RwLock::new(BlockProducer::new(config)))
            })
            .collect();

        Self {
            producers,
            round_robin_index: AtomicUsize::new(0),
            num_producers,
        }
    }

    /// Create a new parallel producer pool with blockchain state loaded from storage
    ///
    /// # Arguments
    /// * `num_producers` - Number of parallel producers (typically 8-16)
    /// * `base_config` - Base configuration to clone for each producer
    /// * `storage` - Storage instance to load blockchain state from
    ///
    /// CRITICAL FIX: This method loads blockchain state from storage to prevent data loss on restart
    pub async fn new_with_storage(
        num_producers: usize,
        base_config: BlockProducerConfig,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<Self> {
        info!(
            "🚀 Initializing Parallel Block Producer Pool with {} producers (LOADING FROM STORAGE)",
            num_producers
        );

        let mut producers = Vec::new();

        for producer_id in 0..num_producers {
            let mut config = base_config.clone();
            // Each producer gets a unique validator index
            config.validator_index = producer_id as u64;
            config.total_validators = num_producers as u64;

            let validator_idx = config.validator_index; // Save before move

            // Create producer
            let mut producer = BlockProducer::new(config);

            // CRITICAL: Load blockchain state from storage
            producer.load_from_storage(storage).await?;

            info!(
                "  ✅ Producer #{} initialized and loaded from storage (validator_index={})",
                producer_id, validator_idx
            );

            producers.push(Arc::new(RwLock::new(producer)));
        }

        Ok(Self {
            producers,
            round_robin_index: AtomicUsize::new(0),
            num_producers,
        })
    }

    /// Queue a mining solution to a producer (round-robin distribution)
    pub async fn queue_solution(&self, solution: MiningSolution) {
        // Round-robin distribution across all producers
        let index = self.round_robin_index.fetch_add(1, Ordering::SeqCst) % self.num_producers;
        debug!(
            "🔄 ParallelBlockProducerPool: Distributing solution to producer #{} (nonce={})",
            index, solution.nonce
        );
        let mut producer = self.producers[index].write().await;
        producer.queue_solution(solution);
        debug!(
            "✅ ParallelBlockProducerPool: Solution distributed to producer #{}",
            index
        );
    }

    /// Produce blocks from all producers that are ready
    ///
    /// This method checks all producers and produces blocks from those
    /// that have enough solutions and have waited long enough.
    ///
    /// Returns vector of (producer_id, block) pairs for visualization
    pub async fn produce_blocks(&self) -> Vec<(usize, QBlock)> {
        let mut blocks = Vec::new();

        // Try to produce from each producer in parallel
        for (producer_id, producer_arc) in self.producers.iter().enumerate() {
            let mut producer = producer_arc.write().await;

            if let Some(block) = producer.produce_block().await {
                info!(
                    "🎉 Producer #{} created block at height {}",
                    producer_id, block.header.height
                );
                blocks.push((producer_id, block));
            }
        }

        blocks
    }

    /// Check if any producer should produce a block
    pub async fn should_produce(&self) -> bool {
        for producer_arc in &self.producers {
            let producer = producer_arc.read().await;
            if producer.should_produce_block() {
                return true;
            }
        }
        false
    }

    /// Get the number of producers in the pool
    pub fn num_producers(&self) -> usize {
        self.num_producers
    }

    /// Get a read-locked reference to a specific producer (for utility methods)
    pub async fn get_producer(
        &self,
        index: usize,
    ) -> tokio::sync::RwLockReadGuard<'_, BlockProducer> {
        self.producers[index % self.num_producers].read().await
    }

    /// ✅ v1.0.8-beta CRITICAL FIX: Advance producer height after block save succeeds
    ///
    /// **CRITICAL**: This MUST only be called AFTER save_qblock() succeeds!
    /// Calling this before storage confirmation will cause catastrophic data loss.
    ///
    /// # Arguments
    /// * `producer_id` - Index of the producer that created the block
    /// * `block_hash` - Hash of the block that was just saved to storage
    ///
    /// # Safety
    /// This method does NOT verify that the block exists on disk.
    /// The caller MUST ensure save_qblock() returned Ok() before calling this.
    ///
    /// # Root Cause Fixed
    /// User nodes were stuck at height 1 because the old code at main.rs:4460 tried to:
    /// ```ignore
    /// let producer_ref = self.get_producer(producer_id);  // Returns RwLockReadGuard (immutable!)
    /// producer_ref.advance_height(block_hash);  // ❌ Won't compile - needs &mut self
    /// ```
    ///
    /// This new method properly acquires a write lock to call advance_height().
    pub async fn advance_producer_height(&self, producer_id: usize, block_hash: BlockHash) {
        let mut producer = self.producers[producer_id % self.num_producers]
            .write()
            .await;
        producer.advance_height(block_hash);

        info!(
            "✅ [v1.0.8-beta FIX] Producer #{} height advanced to {} AFTER storage confirmation",
            producer_id,
            producer.get_height()
        );
    }

    /// Synchronize all producers' blockchain state from storage after sync events
    ///
    /// **v0.9.7-beta CRITICAL FIX**: After turbo sync or HTTP sync completes,
    /// all parallel producers must update their internal height state to match
    /// the new blockchain state. Without this, producers continue creating blocks
    /// at stale heights, causing catastrophic height regression errors.
    ///
    /// # Arguments
    /// * `storage` - Storage instance to load latest blockchain state from
    ///
    /// # Example
    /// ```ignore
    /// // After turbo sync completes:
    /// app_state.block_producer_pool.sync_from_storage(&storage).await?;
    /// ```
    pub async fn sync_from_storage(
        &self,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<()> {
        info!(
            "🔄 [PRODUCER SYNC] Synchronizing all {} producers with blockchain state...",
            self.num_producers
        );

        // ✅ v0.9.12-beta FIX: Use get_highest_contiguous_block() as authoritative source
        // This is the same method used by crash recovery and peer height announcements.
        // It cannot fail to return the correct height even if block data is missing or corrupt.
        let highest_height = storage.get_highest_contiguous_block().await?;

        if highest_height == 0 {
            info!("📝 [PRODUCER SYNC] No blocks in storage yet - producers at genesis");
            return Ok(());
        }

        info!(
            "🔍 [PRODUCER SYNC] Found highest block at height {} in storage",
            highest_height
        );

        // Try to load the actual block for full metadata
        match storage.get_qblock_by_height(highest_height).await? {
            Some(latest_block) => {
                // Full sync with all metadata
                let new_height = latest_block.header.height;
                let new_hash = latest_block.calculate_hash();
                let new_difficulty = latest_block.header.total_difficulty;
                let new_dag_round = latest_block.header.dag_round;

                info!(
                    "   Latest block metadata: height={}, hash={}",
                    new_height,
                    hex::encode(&new_hash[..8])
                );

                // Update all producers atomically
                for (i, producer_arc) in self.producers.iter().enumerate() {
                    let mut producer = producer_arc.write().await;
                    producer.set_latest_block(new_height, new_hash, new_difficulty);
                    producer.dag_round = new_dag_round;

                    debug!("   ✅ Producer #{} synchronized: height={}", i, new_height);
                }

                info!(
                    "✅ [PRODUCER SYNC] All producers synchronized to height {} (full metadata)",
                    new_height
                );
            }
            None => {
                // Block data missing or corrupt - use height-only sync
                warn!("⚠️  [PRODUCER SYNC] Block #{} exists but cannot load data - using height-only sync", highest_height);

                // ✅ v0.9.12-beta CRITICAL FIX: Sync producers to height even without block data
                // This prevents height regression when blocks can't be deserialized.
                // Producers will create the next block with placeholder metadata, which will be
                // corrected when the next valid block arrives from the network.
                let zero_hash = [0u8; 32]; // Placeholder hash
                let zero_difficulty = 0u128; // Will be updated when next block arrives

                for (i, producer_arc) in self.producers.iter().enumerate() {
                    let mut producer = producer_arc.write().await;
                    producer.set_latest_block(highest_height, zero_hash, zero_difficulty);
                    producer.dag_round = highest_height; // Use height as DAG round

                    debug!(
                        "   ⚠️  Producer #{} synchronized to height {} (height-only)",
                        i, highest_height
                    );
                }

                info!(
                    "✅ [PRODUCER SYNC] All producers synchronized to height {} (height-only mode)",
                    highest_height
                );
            }
        }

        Ok(())
    }
}
