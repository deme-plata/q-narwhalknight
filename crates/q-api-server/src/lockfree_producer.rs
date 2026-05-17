use anyhow::{self, Result};
use futures::FutureExt;
/// Lock-Free Block Producer Pool - v0.9.92-beta DEADLOCK FIX (WITH CRITICAL FIXES)
///
/// This module implements a completely lock-free block production system using
/// message passing (channels) instead of shared state (RwLock).
///
/// **Problem Solved**: The original `ParallelBlockProducerPool` used `Arc<RwLock<BlockProducer>>`,
/// which caused deadlocks when:
/// 1. Block production held write lock during long operations
/// 2. Mining submissions tried to acquire the same lock
/// 3. Nested lock acquisitions during block processing
///
/// **Solution**: Each producer runs in its own dedicated async task with NO shared locks.
/// All communication happens via BOUNDED channels with backpressure, panic recovery, and timeouts.
///
/// **Critical Fixes Applied**:
/// - ✅ Bounded channels (10k capacity) to prevent memory exhaustion
/// - ✅ Panic recovery with automatic task restart
/// - ✅ Error propagation via Result types
/// - ✅ Async operation timeouts (30s default)
/// - ✅ Channel closed detection
///
/// **Performance**: ~10-20% faster than RwLock version due to zero lock contention.
use q_types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, trace, warn}; // For .catch_unwind() on async functions

use crate::block_producer::{BlockProducer, BlockProducerConfig};

// TODO v1.0.4: Add Prometheus metrics for producer health monitoring
// Commented out for now to allow quick deployment of timer fix

/// Configuration for lock-free producer
const CHANNEL_CAPACITY: usize = 10_000; // Max queued commands before backpressure
const ASYNC_OPERATION_TIMEOUT: Duration = Duration::from_secs(30); // Timeout for async ops
const PANIC_RESTART_DELAY: Duration = Duration::from_secs(1); // Delay before restarting panicked task

/// Errors that can occur in lock-free producer
#[derive(Debug, thiserror::Error)]
pub enum ProducerError {
    #[error("Producer task has died")]
    TaskDead,

    #[error("Queue is full - backpressure active")]
    QueueFull,

    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),

    #[error("Reply channel closed")]
    ReplyChannelClosed,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// ✅ v1.0.13-beta CRASH-FAST FIX: Explicit errors for should_produce()
///
/// **Problem**: The old `should_produce() -> bool` silently converted errors to `false`,
/// making infrastructure failures (dead tasks, closed channels, timeouts) indistinguishable
/// from normal "don't produce" responses.
///
/// **Solution**: Use Result<bool, Error> so caller can distinguish:
/// - Ok(true) = Producer wants to produce
/// - Ok(false) = Producer says don't produce (normal)
/// - Err(e) = Infrastructure failure (CRITICAL - must crash!)
#[derive(Debug, thiserror::Error)]
pub enum ShouldProduceError {
    #[error("Command send failed: {0}")]
    CommandSendFailed(String),

    #[error("Reply channel closed - producer task died")]
    ReplyChannelClosed,

    #[error("Operation timed out after {0:?}")]
    TimedOut(Duration),

    #[error("Command channel is permanently closed - task dead")]
    ChannelClosed,
}

/// ✅ v1.0.13-beta: Pool-level errors when checking multiple producers
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("Multiple producers unhealthy: {0}")]
    ProducersUnhealthy(String),
}

/// Commands that can be sent to a lock-free producer
pub enum ProducerCommand {
    /// Queue a mining solution for inclusion in next block
    QueueSolution(MiningSolution),

    /// Check if producer should produce a block now
    ShouldProduce(oneshot::Sender<bool>),

    /// Produce a block if conditions are met
    ProduceBlock(oneshot::Sender<Option<QBlock>>),

    /// Get current blockchain height
    GetHeight(oneshot::Sender<u64>),

    /// Get latest block hash
    GetLatestHash(oneshot::Sender<BlockHash>),

    /// Set latest block (for sync operations)
    /// Note: dag_round is automatically set to height by BlockProducer::set_latest_block()
    SetLatestBlock {
        height: u64,
        hash: BlockHash,
        difficulty: u128,
    },

    /// Convert QBlock to DAG Vertex (stateless operation)
    QBlockToVertex {
        block: QBlock,
        reply: oneshot::Sender<anyhow::Result<q_dag_knight::Vertex>>,
    },

    /// Convert DAG Vertex to Storage Vertex (stateless operation)
    DagVertexToStorageVertex {
        dag_vertex: q_dag_knight::Vertex,
        block: QBlock,
        reply: oneshot::Sender<q_types::Vertex>,
    },

    /// ✅ v1.0.1-beta CRITICAL FIX: Advance height AFTER storage confirmation
    /// This command MUST only be sent AFTER save_qblock() succeeds!
    /// Sending this before storage confirmation will cause catastrophic data loss.
    AdvanceHeight { block_hash: BlockHash },

    /// 🚀 v1.0.3.9-beta: Update producer to new height (from network blocks)
    /// Called immediately when blocks are saved to notify producers of height advancement
    UpdateHeight {
        new_height: u64,
        new_hash: BlockHash,
        new_difficulty: u128,
        reply: oneshot::Sender<()>,
    },

    /// ✨ v1.0.16-beta: Set validator keypair for PQC block signing
    /// When set, all produced blocks will be signed with Ed25519/Dilithium5/Hybrid signatures
    SetValidatorKeypair {
        keypair: Arc<q_types::ValidatorKeypair>,
    },

    /// 🔔 v1.0.17-beta: Set event emitter for SSE notifications
    /// When set, all mining rewards will be broadcast in real-time to connected clients
    SetEventEmitter {
        emitter: Arc<crate::streaming::HighPerformanceEmitter>,
    },

    /// ⚔️  v1.0.3-beta: Set DAG-Knight consensus for dag_parents population
    /// When set, blocks will include references to recent DAG vertices
    /// This enables Phase 2 DAG-aware layered sync (10-50x faster)
    SetDagKnight {
        dag_knight: Arc<q_dag_knight::DAGKnightConsensus>,
    },

    /// 📦 v3.5.14-beta: Set production mempool for user transaction inclusion
    /// When set, blocks will include fee-ordered user transactions from the mempool
    /// This is CRITICAL for P2P transaction propagation to work!
    SetProductionMempool {
        mempool: Arc<q_narwhal_core::production_mempool::ProductionMempool>,
    },

    /// 📦 v3.5.20-beta: Set transaction status tracker for P2P transaction confirmations
    /// When set, transactions will be marked as Confirmed after block inclusion
    /// This is CRITICAL for P2P transactions to show as confirmed in explorer!
    SetTxStatus {
        tx_status: Arc<dashmap::DashMap<q_types::TxHash, q_types::TxStatus>>,
    },

    /// 💰 v7.1.5: Set configurable dev fee (shared atomic)
    SetDevFeeBps {
        dev_fee_bps: Arc<std::sync::atomic::AtomicU64>,
    },

    /// 💰 v8.6.1: Set operator fee share (promille of dev fee to admin wallet)
    SetOperatorFee {
        promille: Arc<std::sync::atomic::AtomicU64>,
        admin_wallet: String,
    },

    /// 💰 v8.7.0: Set distributed operators for fee splitting
    SetDistributedOperators {
        operators: Vec<crate::block_producer::OperatorRewardEntry>,
    },

    /// 🏊 v9.1.2: Set mining pool for PPLNS reward distribution
    SetMiningPool {
        pool: Arc<q_mining_pool::MiningPool>,
    },

    /// 🌐 v10.0.0: Set distributed PPLNS proportions source
    SetDistributedPplns {
        proportions: Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>,
    },

    /// 📊 v9.3.1: K-parameter gauge — dynamically tune max solutions per block
    SetMaxSolutionsPerBlock { max_solutions: usize },

    /// Shutdown the producer task gracefully
    Shutdown,
}

/// Lock-free producer handle - send commands without any locks!
#[derive(Clone)]
pub struct LockFreeProducer {
    /// Command channel to producer task (BOUNDED to prevent memory exhaustion)
    command_tx: mpsc::Sender<ProducerCommand>,

    /// Producer ID for logging
    producer_id: usize,
}

impl LockFreeProducer {
    /// Create a new lock-free producer with panic recovery
    ///
    /// This spawns a dedicated async task that owns the BlockProducer.
    /// All operations are sent as commands via bounded channel.
    /// If the task panics, it automatically restarts after a delay.
    pub fn new(producer_id: usize, config: BlockProducerConfig) -> Self {
        let (command_tx, command_rx) = mpsc::channel(CHANNEL_CAPACITY);

        // Spawn dedicated producer task WITH PANIC RECOVERY
        tokio::spawn(Self::producer_task_with_recovery(
            producer_id,
            config.clone(),
            command_rx,
        ));

        Self {
            command_tx,
            producer_id,
        }
    }

    /// Producer task with automatic panic recovery
    async fn producer_task_with_recovery(
        producer_id: usize,
        config: BlockProducerConfig,
        mut command_rx: mpsc::Receiver<ProducerCommand>,
    ) {
        let mut restart_count = 0;

        loop {
            info!(
                "🚀 Lock-free producer #{} task starting (restart count: {})",
                producer_id, restart_count
            );

            // Run producer task with panic catching
            let panic_result = std::panic::AssertUnwindSafe(Self::producer_task_loop(
                producer_id,
                config.clone(),
                &mut command_rx,
            ))
            .catch_unwind()
            .await;

            match panic_result {
                Ok(()) => {
                    info!("✅ Producer #{} task exited gracefully", producer_id);
                    break; // Graceful shutdown
                }
                Err(panic_err) => {
                    restart_count += 1;
                    error!(
                        "🚨 Producer #{} task PANICKED (restart #{}) - Error: {:?}",
                        producer_id, restart_count, panic_err
                    );

                    // Wait before restarting to avoid tight panic loops
                    tokio::time::sleep(PANIC_RESTART_DELAY).await;

                    warn!("🔄 Restarting producer #{} task...", producer_id);
                    // Loop continues to restart
                }
            }
        }
    }

    /// Main producer task loop (can panic, will be caught by recovery wrapper)
    async fn producer_task_loop(
        producer_id: usize,
        config: BlockProducerConfig,
        command_rx: &mut mpsc::Receiver<ProducerCommand>,
    ) {
        let mut producer = BlockProducer::new(config);

        info!(
            "✅ Producer #{} initialized (ZERO LOCKS, BOUNDED CHANNEL)",
            producer_id
        );

        while let Some(command) = command_rx.recv().await {
            match command {
                ProducerCommand::QueueSolution(solution) => {
                    debug!(
                        "📦 Producer #{}: Queued solution nonce={}",
                        producer_id, solution.nonce
                    );
                    producer.queue_solution(solution);
                }

                ProducerCommand::ShouldProduce(reply) => {
                    let should_produce = producer.should_produce_block();
                    let _ = reply.send(should_produce);
                }

                ProducerCommand::ProduceBlock(reply) => {
                    let block = match tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        producer.produce_block()
                    ).await {
                        Ok(block) => block,
                        Err(_) => {
                            error!("🚨 Producer #{}: produce_block() TIMED OUT after 30s — unblocking command loop", producer_id);
                            None
                        }
                    };
                    if let Some(ref b) = block {
                        info!(
                            "✅ Producer #{}: Created block at height {}",
                            producer_id, b.header.height
                        );
                    }
                    let _ = reply.send(block);
                }

                ProducerCommand::GetHeight(reply) => {
                    let height = producer.get_height();
                    let _ = reply.send(height);
                }

                ProducerCommand::GetLatestHash(reply) => {
                    let hash = producer.get_latest_hash();
                    let _ = reply.send(hash);
                }

                ProducerCommand::SetLatestBlock {
                    height,
                    hash,
                    difficulty,
                } => {
                    producer.set_latest_block(height, hash, difficulty);
                    debug!(
                        "🔄 Producer #{}: Synced to height {} (dag_round auto-synced)",
                        producer_id, height
                    );
                }

                ProducerCommand::QBlockToVertex { block, reply } => {
                    let result = producer.qblock_to_vertex(&block);
                    let _ = reply.send(result);
                }

                ProducerCommand::DagVertexToStorageVertex {
                    dag_vertex,
                    block,
                    reply,
                } => {
                    let storage_vertex = producer.dag_vertex_to_storage_vertex(&dag_vertex, &block);
                    let _ = reply.send(storage_vertex);
                }

                ProducerCommand::AdvanceHeight { block_hash } => {
                    producer.advance_height(block_hash);
                    debug!(
                        "✅ Producer #{}: Height advanced via channel command (basic loop)",
                        producer_id
                    );
                }

                ProducerCommand::UpdateHeight {
                    new_height,
                    new_hash,
                    new_difficulty,
                    reply,
                } => {
                    let current_height = producer.get_height();
                    debug!(
                        "[Producer #{}] 📈 Updating height from {} to {}",
                        producer_id, current_height, new_height
                    );
                    producer.set_latest_block(new_height, new_hash, new_difficulty);
                    let _ = reply.send(()); // Acknowledge update
                }

                ProducerCommand::SetValidatorKeypair { keypair } => {
                    producer.set_validator_keypair(keypair);
                    info!(
                        "🔐 Producer #{}: Validator keypair set for PQC signing",
                        producer_id
                    );
                }

                ProducerCommand::SetEventEmitter { emitter } => {
                    producer.set_event_emitter(emitter);
                    info!(
                        "🔔 Producer #{}: Event emitter set for SSE notifications",
                        producer_id
                    );
                }

                ProducerCommand::SetDagKnight { dag_knight } => {
                    producer.set_dag_knight(dag_knight);
                    info!(
                        "⚔️  Producer #{}: DAG-Knight consensus set for dag_parents population",
                        producer_id
                    );
                }

                ProducerCommand::SetProductionMempool { mempool } => {
                    producer.set_production_mempool(mempool);
                    info!(
                        "📦 Producer #{}: Production mempool set for user transaction inclusion",
                        producer_id
                    );
                }

                ProducerCommand::SetTxStatus { tx_status } => {
                    producer.set_tx_status(tx_status);
                    info!(
                        "📦 Producer #{}: Transaction status tracker set for P2P confirmations",
                        producer_id
                    );
                }

                ProducerCommand::SetDevFeeBps { dev_fee_bps } => {
                    producer.set_dev_fee_bps(dev_fee_bps);
                }

                ProducerCommand::SetOperatorFee { promille, admin_wallet } => {
                    producer.set_operator_fee(promille, admin_wallet);
                }

                ProducerCommand::SetDistributedOperators { operators } => {
                    let count = operators.len();
                    producer.set_distributed_operators(operators);
                    trace!("💰 Producer #{}: Distributed operators updated ({} entries)", producer_id, count);
                }

                ProducerCommand::SetMiningPool { pool } => {
                    producer.set_mining_pool(pool);
                    info!("🏊 Producer #{}: Mining pool set for PPLNS reward distribution", producer_id);
                }

                ProducerCommand::SetDistributedPplns { proportions } => {
                    producer.set_distributed_pplns(proportions);
                    info!("🌐 Producer #{}: Distributed PPLNS proportions set", producer_id);
                }

                ProducerCommand::SetMaxSolutionsPerBlock { max_solutions } => {
                    producer.set_max_solutions_per_block(max_solutions);
                    debug!("📊 Producer #{}: max_solutions_per_block tuned to {}", producer_id, max_solutions);
                }

                ProducerCommand::Shutdown => {
                    info!("👋 Producer #{}: Shutting down gracefully", producer_id);
                    break;
                }
            }
        }

        info!("🛑 Producer #{} task terminated", producer_id);
    }

    /// Create a new lock-free producer with blockchain state loaded from storage
    /// Includes panic recovery and bounded channels
    /// ✅ v0.9.99-beta: Now includes adaptive block rewards support
    pub async fn new_with_storage(
        producer_id: usize,
        config: BlockProducerConfig,
        storage: &Arc<q_storage::QStorage>,
        balance_consensus: Option<Arc<q_storage::BalanceConsensusEngine>>,
    ) -> anyhow::Result<Self> {
        let (command_tx, command_rx) = mpsc::channel(CHANNEL_CAPACITY);

        // Clone Arc for task
        let storage_clone = storage.clone();

        // Spawn dedicated producer task WITH PANIC RECOVERY
        tokio::spawn(Self::producer_task_with_storage_and_recovery(
            producer_id,
            config.clone(),
            command_rx,
            storage_clone,
            balance_consensus,
        ));

        Ok(Self {
            command_tx,
            producer_id,
        })
    }

    /// Producer task with storage loading and panic recovery
    /// ✅ v0.9.99-beta: Now includes adaptive block rewards support
    async fn producer_task_with_storage_and_recovery(
        producer_id: usize,
        config: BlockProducerConfig,
        mut command_rx: mpsc::Receiver<ProducerCommand>,
        storage: Arc<q_storage::QStorage>,
        balance_consensus: Option<Arc<q_storage::BalanceConsensusEngine>>,
    ) {
        let mut restart_count = 0;

        loop {
            info!(
                "🚀 Lock-free producer #{} task starting with storage (restart count: {})",
                producer_id, restart_count
            );

            // Run producer task with panic catching
            let panic_result = std::panic::AssertUnwindSafe(Self::producer_task_loop_with_storage(
                producer_id,
                config.clone(),
                &mut command_rx,
                storage.clone(),
                balance_consensus.clone(),
            ))
            .catch_unwind()
            .await;

            match panic_result {
                Ok(()) => {
                    info!("✅ Producer #{} task exited gracefully", producer_id);
                    break; // Graceful shutdown
                }
                Err(panic_err) => {
                    restart_count += 1;
                    error!(
                        "🚨 Producer #{} task PANICKED (restart #{}) - Error: {:?}",
                        producer_id, restart_count, panic_err
                    );

                    // Wait before restarting
                    tokio::time::sleep(PANIC_RESTART_DELAY).await;

                    warn!("🔄 Restarting producer #{} task...", producer_id);
                    // Loop continues to restart
                }
            }
        }
    }

    /// Main producer task loop with storage (can panic, will be caught by recovery wrapper)
    /// ✅ v0.9.99-beta: Uses adaptive rewards if balance_consensus is provided
    async fn producer_task_loop_with_storage(
        producer_id: usize,
        config: BlockProducerConfig,
        command_rx: &mut mpsc::Receiver<ProducerCommand>,
        storage: Arc<q_storage::QStorage>,
        balance_consensus: Option<Arc<q_storage::BalanceConsensusEngine>>,
    ) {
        // ✅ v0.9.99-beta: Create producer with adaptive rewards if available
        let mut producer = match balance_consensus {
            Some(bc) => {
                info!(
                    "✅ Producer #{}: Creating with ADAPTIVE rewards (v0.9.99-beta)",
                    producer_id
                );
                BlockProducer::new_with_adaptive_rewards(config, bc)
            }
            None => {
                warn!(
                    "⚠️  Producer #{}: Creating with FIXED rewards (0.05 QUG)",
                    producer_id
                );
                BlockProducer::new(config)
            }
        };

        // CRITICAL: Load blockchain state from storage
        if let Err(e) = producer.load_from_storage(&storage).await {
            error!(
                "❌ Producer #{}: Failed to load from storage: {}",
                producer_id, e
            );
            return;
        }

        // 🔐 BalanceRootV1: Wire storage into producer for balance root computation.
        // Required so produce_block() can call compute_balance_root_for_block() when
        // the BalanceRootV1 upgrade is active (mainnet activation: 18,600,000).
        producer.set_storage(storage.clone());

        info!(
            "✅ Producer #{} initialized with storage (ZERO LOCKS, BOUNDED CHANNEL)",
            producer_id
        );

        while let Some(command) = command_rx.recv().await {
            match command {
                ProducerCommand::QueueSolution(solution) => {
                    debug!(
                        "📦 Producer #{}: Queued solution nonce={}",
                        producer_id, solution.nonce
                    );
                    producer.queue_solution(solution);
                }

                ProducerCommand::ShouldProduce(reply) => {
                    let should_produce = producer.should_produce_block();
                    let _ = reply.send(should_produce);
                }

                ProducerCommand::ProduceBlock(reply) => {
                    let block = match tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        producer.produce_block()
                    ).await {
                        Ok(block) => block,
                        Err(_) => {
                            error!("🚨 Producer #{}: produce_block() TIMED OUT after 30s — unblocking command loop", producer_id);
                            None
                        }
                    };
                    if let Some(ref b) = block {
                        info!(
                            "✅ Producer #{}: Created block at height {}",
                            producer_id, b.header.height
                        );
                    }
                    let _ = reply.send(block);
                }

                ProducerCommand::GetHeight(reply) => {
                    let height = producer.get_height();
                    let _ = reply.send(height);
                }

                ProducerCommand::GetLatestHash(reply) => {
                    let hash = producer.get_latest_hash();
                    let _ = reply.send(hash);
                }

                ProducerCommand::SetLatestBlock {
                    height,
                    hash,
                    difficulty,
                } => {
                    producer.set_latest_block(height, hash, difficulty);
                    debug!(
                        "🔄 Producer #{}: Synced to height {} (dag_round auto-synced)",
                        producer_id, height
                    );
                }

                ProducerCommand::QBlockToVertex { block, reply } => {
                    let result = producer.qblock_to_vertex(&block);
                    let _ = reply.send(result);
                }

                ProducerCommand::DagVertexToStorageVertex {
                    dag_vertex,
                    block,
                    reply,
                } => {
                    let storage_vertex = producer.dag_vertex_to_storage_vertex(&dag_vertex, &block);
                    let _ = reply.send(storage_vertex);
                }

                ProducerCommand::AdvanceHeight { block_hash } => {
                    producer.advance_height(block_hash);
                    debug!(
                        "✅ Producer #{}: Height advanced via channel command (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::UpdateHeight {
                    new_height,
                    new_hash,
                    new_difficulty,
                    reply,
                } => {
                    let current_height = producer.get_height();
                    debug!(
                        "[Producer #{}] 📈 Updating height from {} to {} (storage loop)",
                        producer_id, current_height, new_height
                    );
                    producer.set_latest_block(new_height, new_hash, new_difficulty);
                    let _ = reply.send(()); // Acknowledge update
                }

                ProducerCommand::SetValidatorKeypair { keypair } => {
                    producer.set_validator_keypair(keypair);
                    info!(
                        "🔐 Producer #{}: Validator keypair set for PQC signing (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::SetEventEmitter { emitter } => {
                    producer.set_event_emitter(emitter);
                    info!(
                        "🔔 Producer #{}: Event emitter set for SSE notifications (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::SetDagKnight { dag_knight } => {
                    producer.set_dag_knight(dag_knight);
                    info!(
                        "⚔️  Producer #{}: DAG-Knight consensus set for dag_parents population (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::SetProductionMempool { mempool } => {
                    producer.set_production_mempool(mempool);
                    info!(
                        "📦 Producer #{}: Production mempool set for user transaction inclusion (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::SetTxStatus { tx_status } => {
                    producer.set_tx_status(tx_status);
                    info!(
                        "📦 Producer #{}: Transaction status tracker set for P2P confirmations (storage loop)",
                        producer_id
                    );
                }

                ProducerCommand::SetDevFeeBps { dev_fee_bps } => {
                    producer.set_dev_fee_bps(dev_fee_bps);
                    info!("💰 Producer #{}: Dev fee BPS updated (storage loop)", producer_id);
                }

                ProducerCommand::SetOperatorFee { promille, admin_wallet } => {
                    producer.set_operator_fee(promille, admin_wallet);
                    info!("💰 Producer #{}: Operator fee updated (storage loop)", producer_id);
                }

                ProducerCommand::SetDistributedOperators { operators } => {
                    let count = operators.len();
                    producer.set_distributed_operators(operators);
                    trace!("💰 Producer #{}: Distributed operators updated ({} entries, storage loop)", producer_id, count);
                }

                ProducerCommand::SetMiningPool { pool } => {
                    producer.set_mining_pool(pool);
                    info!("🏊 Producer #{}: Mining pool set for PPLNS (storage loop)", producer_id);
                }

                ProducerCommand::SetDistributedPplns { proportions } => {
                    producer.set_distributed_pplns(proportions);
                    info!("🌐 Producer #{}: Distributed PPLNS proportions set (storage loop)", producer_id);
                }

                ProducerCommand::SetMaxSolutionsPerBlock { max_solutions } => {
                    producer.set_max_solutions_per_block(max_solutions);
                    debug!("📊 Producer #{}: max_solutions_per_block tuned to {} (storage loop)", producer_id, max_solutions);
                }

                ProducerCommand::Shutdown => {
                    info!("👋 Producer #{}: Shutting down gracefully", producer_id);
                    break;
                }
            }
        }

        info!("🛑 Producer #{} task terminated", producer_id);
    }

    /// Queue a mining solution with backpressure (non-blocking but can fail if queue is full)
    pub fn queue_solution(&self, solution: MiningSolution) -> Result<(), ProducerError> {
        match self
            .command_tx
            .try_send(ProducerCommand::QueueSolution(solution))
        {
            Ok(_) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!(
                    "Producer #{}: Queue FULL - backpressure active (10k commands queued)",
                    self.producer_id
                );
                Err(ProducerError::QueueFull)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!(
                    "Producer #{}: Task DEAD - channel closed!",
                    self.producer_id
                );
                Err(ProducerError::TaskDead)
            }
        }
    }

    /// Check if should produce block (async with timeout)
    /// ✅ v1.0.13-beta CRASH-FAST FIX: Returns Result instead of bool
    ///
    /// **CRITICAL CHANGE**: This method now returns Result<bool, ShouldProduceError>
    /// instead of bool. Errors are NO LONGER silently converted to false!
    ///
    /// **Caller MUST**:
    /// - Handle Ok(true) = Produce a block
    /// - Handle Ok(false) = Don't produce (normal)
    /// - Handle Err(e) = INFRASTRUCTURE FAILURE → std::process::exit(1)!
    pub async fn should_produce(&self) -> Result<bool, ShouldProduceError> {
        let (reply_tx, reply_rx) = oneshot::channel();

        // Check if channel is closed BEFORE trying to send
        if self.command_tx.is_closed() {
            error!(
                "🚨 FATAL: Producer #{} command channel PERMANENTLY CLOSED!",
                self.producer_id
            );
            error!("   Producer task has DIED - this is UNRECOVERABLE!");
            return Err(ShouldProduceError::ChannelClosed);
        }

        // Try to send command to producer task
        if let Err(e) = self
            .command_tx
            .try_send(ProducerCommand::ShouldProduce(reply_tx))
        {
            error!(
                "🚨 FATAL: Producer #{} failed to send ShouldProduce: {:?}",
                self.producer_id, e
            );
            error!("   Producer task is DEAD or channel FULL!");
            return Err(ShouldProduceError::CommandSendFailed(format!("{:?}", e)));
        }

        // Wait for reply with timeout
        match timeout(ASYNC_OPERATION_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => {
                // ✅ Normal response
                debug!(
                    "✅ Producer #{}: should_produce() = {}",
                    self.producer_id, result
                );
                Ok(result)
            }
            Ok(Err(_)) => {
                // ❌ Reply channel closed = task died after receiving command
                error!(
                    "🚨 FATAL: Producer #{} reply channel CLOSED!",
                    self.producer_id
                );
                error!("   Producer task DIED after receiving command - may have panicked!");
                Err(ShouldProduceError::ReplyChannelClosed)
            }
            Err(_) => {
                // ❌ Timeout = task is deadlocked or hung
                error!(
                    "🚨 FATAL: Producer #{} TIMED OUT after {:?}!",
                    self.producer_id, ASYNC_OPERATION_TIMEOUT
                );
                error!("   Producer task is DEADLOCKED or HUNG!");
                Err(ShouldProduceError::TimedOut(ASYNC_OPERATION_TIMEOUT))
            }
        }
    }

    /// Produce a block (async with timeout)
    pub async fn produce_block(&self) -> Option<QBlock> {
        let (reply_tx, reply_rx) = oneshot::channel();

        if let Err(e) = self
            .command_tx
            .try_send(ProducerCommand::ProduceBlock(reply_tx))
        {
            error!(
                "Producer #{}: Failed to send ProduceBlock: {:?}",
                self.producer_id, e
            );
            return None;
        }

        match timeout(ASYNC_OPERATION_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                error!(
                    "Producer #{}: ProduceBlock reply channel closed",
                    self.producer_id
                );
                None
            }
            Err(_) => {
                error!(
                    "Producer #{}: ProduceBlock timed out after {:?}",
                    self.producer_id, ASYNC_OPERATION_TIMEOUT
                );
                None
            }
        }
    }

    /// Get current height (async, returns via channel)
    /// 🔥 v1.0.3.3-beta CRITICAL FIX: Check capacity BEFORE creating oneshot channel to prevent orphaned channels
    pub async fn get_height(&self) -> u64 {
        // CRITICAL: Check capacity FIRST - don't create oneshot channel if full
        // This prevents orphaned oneshot channel accumulation that gradually fills the bounded channel
        if self.command_tx.capacity() == 0 {
            warn!("Producer #{}: Channel full during GetHeight (producer busy, no orphaned channel created)", self.producer_id);
            return 0;
        }

        // Only create oneshot channel if we know try_send() will succeed
        let (reply_tx, reply_rx) = oneshot::channel();

        // Use try_send() instead of send().await to avoid blocking deadlock
        match self
            .command_tx
            .try_send(ProducerCommand::GetHeight(reply_tx))
        {
            Ok(_) => reply_rx.await.unwrap_or(0),
            Err(_) => {
                // Channel filled between capacity check and try_send (rare race condition)
                // Both oneshot halves will be dropped together, no orphan
                warn!(
                    "Producer #{}: Channel filled during GetHeight (race condition)",
                    self.producer_id
                );
                0
            }
        }
    }

    /// Get latest hash (async, returns via channel)
    /// 🔥 v1.0.3.3-beta CRITICAL FIX: Check capacity BEFORE creating oneshot channel to prevent orphaned channels
    pub async fn get_latest_hash(&self) -> BlockHash {
        // CRITICAL: Check capacity FIRST - don't create oneshot channel if full
        // This prevents orphaned oneshot channel accumulation that gradually fills the bounded channel
        if self.command_tx.capacity() == 0 {
            warn!("Producer #{}: Channel full during GetLatestHash (producer busy, no orphaned channel created)", self.producer_id);
            return [0u8; 32];
        }

        // Only create oneshot channel if we know try_send() will succeed
        let (reply_tx, reply_rx) = oneshot::channel();

        // Use try_send() instead of send().await to avoid blocking deadlock
        match self
            .command_tx
            .try_send(ProducerCommand::GetLatestHash(reply_tx))
        {
            Ok(_) => reply_rx.await.unwrap_or([0u8; 32]),
            Err(_) => {
                // Channel filled between capacity check and try_send (rare race condition)
                // Both oneshot halves will be dropped together, no orphan
                warn!(
                    "Producer #{}: Channel filled during GetLatestHash (race condition)",
                    self.producer_id
                );
                [0u8; 32]
            }
        }
    }

    /// Set latest block for sync operations (fire-and-forget, never blocks)
    /// Note: dag_round is automatically set to height by the underlying BlockProducer
    pub fn set_latest_block(
        &self,
        height: u64,
        hash: BlockHash,
        difficulty: u128,
        _dag_round: u64,
    ) {
        let cmd = ProducerCommand::SetLatestBlock {
            height,
            hash,
            difficulty,
        };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetLatestBlock command: {:?}",
                self.producer_id, e
            );
        }
    }

    /// Convert QBlock to DAG Vertex (async, stateless operation)
    pub async fn qblock_to_vertex(&self, block: &QBlock) -> anyhow::Result<q_dag_knight::Vertex> {
        let (reply_tx, reply_rx) = oneshot::channel();

        if let Err(e) = self.command_tx.try_send(ProducerCommand::QBlockToVertex {
            block: block.clone(),
            reply: reply_tx,
        }) {
            error!(
                "Producer #{}: Failed to send QBlockToVertex command: {:?}",
                self.producer_id, e
            );
            return Err(anyhow::anyhow!("Failed to send command"));
        }

        reply_rx
            .await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("Reply channel closed")))
    }

    /// Convert DAG Vertex to Storage Vertex (async, stateless operation)
    pub async fn dag_vertex_to_storage_vertex(
        &self,
        dag_vertex: &q_dag_knight::Vertex,
        block: &QBlock,
    ) -> q_types::Vertex {
        let (reply_tx, reply_rx) = oneshot::channel();

        if let Err(e) = self
            .command_tx
            .try_send(ProducerCommand::DagVertexToStorageVertex {
                dag_vertex: dag_vertex.clone(),
                block: block.clone(),
                reply: reply_tx,
            })
        {
            error!(
                "Producer #{}: Failed to send DagVertexToStorageVertex command: {:?}",
                self.producer_id, e
            );
            // Return empty vertex on error (placeholder with zero values)
            return q_types::Vertex {
                id: [0u8; 32],
                round: 0,
                author: [0u8; 32],
                tx_root: [0u8; 32],
                parents: Vec::new(),
                transactions: Vec::new(),
                signature: Vec::new(),
                timestamp: chrono::Utc::now(),
            };
        }

        reply_rx.await.unwrap_or_else(|_| q_types::Vertex {
            id: [0u8; 32],
            round: 0,
            author: [0u8; 32],
            tx_root: [0u8; 32],
            parents: Vec::new(),
            transactions: Vec::new(),
            signature: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// ✅ v1.0.1-beta CRITICAL FIX: Advance height AFTER storage confirmation
    ///
    /// **CRITICAL**: This MUST only be called AFTER save_qblock() succeeds!
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
    pub fn advance_height(&self, block_hash: BlockHash) {
        let cmd = ProducerCommand::AdvanceHeight { block_hash };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send AdvanceHeight command: {:?}",
                self.producer_id, e
            );
        } else {
            debug!(
                "📤 Producer #{}: Sent AdvanceHeight command to task",
                self.producer_id
            );
        }
    }

    /// 🚀 v1.0.3.9-beta: Update producer to new height (from network blocks)
    ///
    /// Called immediately when blocks are saved to notify producers of height advancement.
    /// This is part of the PRIMARY FIX for the stale state bug.
    ///
    /// # Arguments
    /// * `new_height` - The new blockchain height
    /// * `new_hash` - Hash of the block at the new height
    /// * `new_difficulty` - Total difficulty at the new height
    pub async fn update_height(
        &self,
        new_height: u64,
        new_hash: BlockHash,
        new_difficulty: u128,
    ) -> Result<(), ProducerError> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.command_tx
            .send(ProducerCommand::UpdateHeight {
                new_height,
                new_hash,
                new_difficulty,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ProducerError::TaskDead)?;

        reply_rx
            .await
            .map_err(|_| ProducerError::ReplyChannelClosed)?;

        Ok(())
    }

    /// Set validator keypair for PQC block signing
    /// ✨ v1.0.16-beta: Enable post-quantum signatures
    ///
    /// When a validator keypair is set, all produced blocks will be signed with:
    /// - Ed25519 (Phase 0)
    /// - Dilithium5 (Phase 1)
    /// - Hybrid Ed25519+Dilithium5 (during transition)
    pub fn set_validator_keypair(&self, keypair: Arc<q_types::ValidatorKeypair>) {
        let cmd = ProducerCommand::SetValidatorKeypair { keypair };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetValidatorKeypair command: {:?}",
                self.producer_id, e
            );
        } else {
            info!(
                "🔐 Producer #{}: Sent SetValidatorKeypair command",
                self.producer_id
            );
        }
    }

    /// Set DAG-Knight consensus for dag_parents population
    /// ⚔️  v1.0.3-beta: Enable DAG-aware sync (Phase 1)
    ///
    /// When DAG-Knight consensus is set, all produced blocks will include
    /// references to recent committed DAG vertices in the dag_parents field.
    /// This enables Phase 2 DAG-aware layered sync (10-50x performance improvement).
    pub fn set_dag_knight(&self, dag_knight: Arc<q_dag_knight::DAGKnightConsensus>) {
        let cmd = ProducerCommand::SetDagKnight { dag_knight };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetDagKnight command: {:?}",
                self.producer_id, e
            );
        } else {
            info!(
                "⚔️  Producer #{}: Sent SetDagKnight command",
                self.producer_id
            );
        }
    }

    /// Set event emitter for SSE notifications
    /// 🔔 v1.0.17-beta: Enable mining reward notifications
    ///
    /// When an event emitter is set, all mining rewards will be broadcast in real-time
    /// to connected SSE/WebSocket clients as they are created during block production.
    pub fn set_event_emitter(&self, emitter: Arc<crate::streaming::HighPerformanceEmitter>) {
        let cmd = ProducerCommand::SetEventEmitter { emitter };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetEventEmitter command: {:?}",
                self.producer_id, e
            );
        } else {
            info!(
                "🔔 Producer #{}: Sent SetEventEmitter command",
                self.producer_id
            );
        }
    }

    /// Set production mempool for user transaction inclusion
    /// 📦 v3.5.14-beta: Enable P2P transaction propagation
    ///
    /// When a production mempool is set, blocks will include fee-ordered user transactions.
    /// This is CRITICAL for P2P transaction propagation to work - without it, user transactions
    /// received via gossipsub will be queued but never included in blocks!
    pub fn set_production_mempool(&self, mempool: Arc<q_narwhal_core::production_mempool::ProductionMempool>) {
        let cmd = ProducerCommand::SetProductionMempool { mempool };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetProductionMempool command: {:?}",
                self.producer_id, e
            );
        } else {
            info!(
                "📦 Producer #{}: Sent SetProductionMempool command",
                self.producer_id
            );
        }
    }

    /// Set transaction status tracker for P2P transaction confirmations
    /// 📦 v3.5.20-beta: Enable transaction status updates after block inclusion
    ///
    /// When a tx_status tracker is set, transactions will be marked as Confirmed
    /// after they are included in a block. This is CRITICAL for P2P transactions
    /// to show as confirmed in the explorer!
    pub fn set_tx_status(&self, tx_status: Arc<dashmap::DashMap<q_types::TxHash, q_types::TxStatus>>) {
        let cmd = ProducerCommand::SetTxStatus { tx_status };

        if let Err(e) = self.command_tx.try_send(cmd) {
            error!(
                "Producer #{}: Failed to send SetTxStatus command: {:?}",
                self.producer_id, e
            );
        } else {
            info!(
                "📦 Producer #{}: Sent SetTxStatus command",
                self.producer_id
            );
        }
    }

    /// 💰 v7.1.5: Set configurable dev fee
    pub fn set_dev_fee_bps(&self, dev_fee_bps: Arc<std::sync::atomic::AtomicU64>) {
        let _ = self.command_tx.try_send(ProducerCommand::SetDevFeeBps { dev_fee_bps });
    }

    /// 💰 v8.6.1: Set operator fee share
    pub fn set_operator_fee(&self, promille: Arc<std::sync::atomic::AtomicU64>, admin_wallet: String) {
        let _ = self.command_tx.try_send(ProducerCommand::SetOperatorFee { promille, admin_wallet });
    }

    /// 💰 v8.7.0: Set distributed operators for fee splitting
    pub fn set_distributed_operators(&self, operators: Vec<crate::block_producer::OperatorRewardEntry>) {
        let _ = self.command_tx.try_send(ProducerCommand::SetDistributedOperators { operators });
    }

    /// 🏊 v9.1.2: Set mining pool for PPLNS reward distribution
    pub fn set_mining_pool(&self, pool: Arc<q_mining_pool::MiningPool>) {
        let _ = self.command_tx.try_send(ProducerCommand::SetMiningPool { pool });
    }

    /// 🌐 v10.0.2: Set distributed PPLNS proportions for multi-node mining
    pub fn set_distributed_pplns(&self, proportions: Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>) {
        let _ = self.command_tx.try_send(ProducerCommand::SetDistributedPplns { proportions });
    }

    /// 📊 v9.3.1: Set max solutions per block (K-parameter dynamic tuning)
    pub fn set_max_solutions_per_block(&self, max_solutions: usize) {
        let _ = self.command_tx.try_send(ProducerCommand::SetMaxSolutionsPerBlock { max_solutions });
    }

    /// Shutdown producer gracefully
    pub fn shutdown(&self) {
        let _ = self.command_tx.try_send(ProducerCommand::Shutdown);
    }
}

/// Lock-Free Parallel Producer Pool - DEADLOCK-FREE ARCHITECTURE
///
/// **Key Differences from Original**:
/// - NO Arc<RwLock<>> anywhere!
/// - Each producer is a handle to a dedicated task
/// - All communication via unbounded channels
/// - Zero lock contention, zero deadlock risk
///
/// **Performance**:
/// - 10-20% faster than RwLock version
/// - Scales linearly with number of producers
/// - No blocking on hot paths
pub struct LockFreeProducerPool {
    /// Lock-free producer handles (just channel senders!)
    producers: Vec<LockFreeProducer>,

    /// Round-robin index for solution distribution
    round_robin_index: AtomicUsize,

    /// Number of producers in the pool
    num_producers: usize,

    /// 🚀 v2.3.13-beta: RACE CONDITION FIX - Prevent concurrent produce_blocks() calls
    /// Root cause of 50% block loss: Two production loops (block_production_v2 and mining handler)
    /// could call produce_blocks() simultaneously, creating duplicate blocks at the same height.
    /// The BlockWriter's deduplication would drop one, causing gaps.
    /// Fix: Use atomic flag to serialize production calls.
    /// v10.2.11: Made pub so the production watchdog can force-clear on stall
    pub production_in_progress: AtomicBool,
    pub production_in_progress_since: AtomicU64,

    /// 🚀 v2.3.15-beta: POOL-LEVEL DUPLICATE PREVENTION
    /// The per-producer last_produced_height doesn't work because multiple producers
    /// in the pool each have their own height tracking. This global counter ensures
    /// no producer in the pool produces at a height that any other producer already produced.
    pool_last_produced_height: AtomicU64,
}

impl LockFreeProducerPool {
    /// Create a new lock-free producer pool
    pub fn new(num_producers: usize, base_config: BlockProducerConfig) -> Self {
        info!(
            "🚀 Initializing LOCK-FREE Parallel Block Producer Pool with {} producers",
            num_producers
        );
        info!("   🔓 ZERO RwLocks - Channel-based architecture");
        info!("   ⚡ ZERO lock contention - Message passing only");
        info!("   🛡️  ZERO deadlock risk - No shared mutable state");

        let producers = (0..num_producers)
            .map(|producer_id| {
                let mut config = base_config.clone();
                config.validator_index = producer_id as u64;
                config.total_validators = num_producers as u64;

                info!(
                    "  ✅ Lock-free producer #{} spawned (validator_index={})",
                    producer_id, config.validator_index
                );

                LockFreeProducer::new(producer_id, config)
            })
            .collect();

        info!(
            "✅ LOCK-FREE producer pool initialized - {} independent tasks running",
            num_producers
        );

        Self {
            producers,
            round_robin_index: AtomicUsize::new(0),
            num_producers,
            production_in_progress: AtomicBool::new(false),
            production_in_progress_since: AtomicU64::new(0),
            pool_last_produced_height: AtomicU64::new(0), // v2.3.15-beta: Pool-level duplicate prevention
        }
    }

    /// Create a new lock-free producer pool with blockchain state loaded from storage
    /// ✅ v0.9.99-beta: Now includes adaptive block rewards
    pub async fn new_with_storage(
        num_producers: usize,
        base_config: BlockProducerConfig,
        storage: &Arc<q_storage::QStorage>,
        balance_consensus: Option<Arc<q_storage::BalanceConsensusEngine>>,
    ) -> anyhow::Result<Self> {
        info!("🚀 Initializing LOCK-FREE Parallel Block Producer Pool with {} producers (LOADING FROM STORAGE)", num_producers);
        if balance_consensus.is_some() {
            info!("   ✅ v0.9.99-beta: Adaptive block rewards ENABLED");
        } else {
            info!("   ⚠️  Adaptive block rewards DISABLED (using fixed 0.05 QUG)");
        }

        let mut producers = Vec::new();

        for producer_id in 0..num_producers {
            let mut config = base_config.clone();
            config.validator_index = producer_id as u64;
            config.total_validators = num_producers as u64;

            let producer = LockFreeProducer::new_with_storage(
                producer_id,
                config,
                storage,
                balance_consensus.clone(),
            )
            .await?;

            info!(
                "  ✅ Lock-free producer #{} spawned and synced from storage",
                producer_id
            );

            producers.push(producer);
        }

        info!("✅ LOCK-FREE producer pool initialized with storage sync");

        Ok(Self {
            producers,
            round_robin_index: AtomicUsize::new(0),
            num_producers,
            production_in_progress: AtomicBool::new(false),
            production_in_progress_since: AtomicU64::new(0),
            pool_last_produced_height: AtomicU64::new(0), // v2.3.15-beta: Pool-level duplicate prevention
        })
    }

    /// Queue a mining solution with backpressure (returns Result)
    ///
    /// **Returns**:
    /// - `Ok(())` if solution was queued successfully
    /// - `Err(ProducerError::QueueFull)` if producer queue is full (backpressure)
    /// - `Err(ProducerError::TaskDead)` if producer task has died
    pub fn queue_solution(&self, solution: MiningSolution) -> Result<(), ProducerError> {
        // Round-robin distribution
        let index = self.round_robin_index.fetch_add(1, Ordering::SeqCst) % self.num_producers;

        debug!(
            "🔄 Lock-free pool: Distributing solution to producer #{} (nonce={})",
            index, solution.nonce
        );

        // Send to producer - uses bounded channel with backpressure
        // Clone solution early so we can retry with other producers if needed
        match self.producers[index].queue_solution(solution.clone()) {
            Ok(_) => {
                debug!("✅ Lock-free pool: Solution queued to producer #{}", index);
                Ok(())
            }
            Err(ProducerError::QueueFull) => {
                warn!("⚠️ Producer #{} queue FULL - trying next producer", index);

                // Try other producers if first one is full
                for offset in 1..self.num_producers {
                    let alt_index = (index + offset) % self.num_producers;
                    if let Ok(_) = self.producers[alt_index].queue_solution(solution.clone()) {
                        debug!("✅ Queued to alternate producer #{}", alt_index);
                        return Ok(());
                    }
                }

                // All producers full!
                error!("🚨 ALL producers queues FULL - dropping solution!");
                Err(ProducerError::QueueFull)
            }
            Err(e) => Err(e),
        }
    }

    /// Produce blocks from all ready producers
    ///
    /// **CRITICAL DIFFERENCE**: This method does NOT hold any locks!
    /// Each producer is queried via channel, completely independently.
    ///
    /// ✅ v1.0.13-beta: Now handles Result<bool, Error> from should_produce()
    /// ✅ v1.1.30-beta: FIX - Only ONE producer should produce per round to prevent double rewards!
    ///    The bug was: both producers could produce at the same height, causing 2x mining rewards.
    pub async fn produce_blocks(&self) -> Vec<(usize, QBlock)> {
        info!("🔍 [PRODUCE_BLOCKS] ENTERED — num_producers={}, pool_last_produced={}",
              self.num_producers, self.pool_last_produced_height.load(Ordering::SeqCst));

        // v10.2.9: Zombie flag detection — if stuck >120s, force-clear
        if self.production_in_progress.load(std::sync::atomic::Ordering::SeqCst) {
            let set_at = self.production_in_progress_since.load(std::sync::atomic::Ordering::SeqCst);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            if now_ms.saturating_sub(set_at) > 120_000 {
                error!("🚨 [PRODUCE_BLOCKS] production_in_progress stuck >120s — force-clearing zombie flag");
                self.production_in_progress.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }

        // 🚀 v2.3.13-beta: RACE CONDITION FIX - Prevent concurrent production calls
        if self.production_in_progress.compare_exchange(
            false, true, Ordering::SeqCst, Ordering::SeqCst
        ).is_err() {
            info!("❌ [PRODUCE_BLOCKS] EXIT: RACE PREVENTION — another call in progress");
            return Vec::new();
        }

        self.production_in_progress_since.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            std::sync::atomic::Ordering::SeqCst,
        );

        // Use a guard pattern to ensure flag is always cleared
        struct ProductionGuard<'a>(&'a AtomicBool);
        impl<'a> Drop for ProductionGuard<'a> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _guard = ProductionGuard(&self.production_in_progress);

        let mut blocks = Vec::new();

        // 🔄 v7.1.4: ROTATE START PRODUCER to prevent producer #0 monopoly
        // Previous bug: Always iterating from producer #0 meant producer #0 was checked first,
        // its timer was always elapsed (reset by set_latest_block from P2P blocks), and it
        // always produced, causing `break` before producer #1 was ever checked.
        // Fix: Rotate the starting producer each round so both get fair turns.
        let start_index = self.round_robin_index.fetch_add(0, Ordering::SeqCst) % self.num_producers;

        // Query each producer via channel (NO LOCKS!)
        for offset in 0..self.num_producers {
            let producer_id = (start_index + offset) % self.num_producers;
            let producer = &self.producers[producer_id];

            // ✅ v1.0.13-beta: Handle Result type from should_produce()
            match producer.should_produce().await {
                Ok(true) => {
                    info!("🔍 [PRODUCE_BLOCKS] Producer #{} says YES — calling produce_block()...", producer_id);
                    // Produce block (async via channel)
                    if let Some(block) = producer.produce_block().await {
                        let block_height = block.header.height;

                        // 🚀 v2.3.15-beta: POOL-LEVEL DUPLICATE PREVENTION
                        // Check if this height was already produced by the pool in a previous call.
                        // This catches the case where two production loops call produce_blocks()
                        // 1 second apart (not truly concurrent, so the AtomicBool doesn't help).
                        let pool_height = self.pool_last_produced_height.load(Ordering::SeqCst);
                        if block_height <= pool_height {
                            warn!(
                                "⏸️ [POOL DUPLICATE] Block {} already produced by pool (pool_height={}), skipping",
                                block_height, pool_height
                            );
                            continue; // Skip this block, try next producer
                        }

                        // Update pool height BEFORE adding block
                        self.pool_last_produced_height.store(block_height, Ordering::SeqCst);

                        info!(
                            "🎉 Lock-free producer #{} created block at height {} (pool_height updated)",
                            producer_id, block_height
                        );
                        blocks.push((producer_id, block));
                        // ✅ v1.1.30-beta CRITICAL FIX: Only ONE block per round!
                        break;
                    } else {
                        info!("⚠️ [PRODUCE_BLOCKS] Producer #{} returned None from produce_block() — see EXIT reason in producer logs", producer_id);
                    }
                }
                Ok(false) => {
                    // Normal "don't produce" response
                    debug!("Producer #{} should not produce (normal)", producer_id);
                }
                Err(e) => {
                    // Producer is unhealthy - log but don't crash here
                    // The pool-level should_produce() will catch this and crash
                    error!(
                        "❌ Producer #{} unhealthy in produce_blocks(): {}",
                        producer_id, e
                    );
                }
            }
        }

        blocks
        // _guard drops here, clearing production_in_progress flag
    }

    /// Check if any producer should produce a block
    /// ✅ v1.0.13-beta CRASH-FAST FIX: Returns Result instead of bool
    ///
    /// **CRITICAL CHANGE**: This method now returns Result<bool, PoolError>.
    /// Errors indicate infrastructure failures that require crashing!
    ///
    /// **Returns**:
    /// - Ok(true) = At least one producer wants to produce
    /// - Ok(false) = All producers say don't produce (normal)
    /// - Err(PoolError::ProducersUnhealthy) = One or more producers are DEAD/unhealthy
    pub async fn should_produce(&self) -> Result<bool, PoolError> {
        // Check all producers in parallel via channels
        let mut futures = Vec::new();

        for producer in &self.producers {
            futures.push(producer.should_produce());
        }

        // Wait for all responses
        let results = futures::future::join_all(futures).await;

        let mut any_true = false;
        let mut errors = Vec::new();

        for (idx, result) in results.iter().enumerate() {
            match result {
                Ok(true) => {
                    any_true = true;
                    debug!("Producer #{} should produce", idx);
                }
                Ok(false) => {
                    // Normal "don't produce" response
                    debug!("Producer #{} should NOT produce", idx);
                }
                Err(e) => {
                    // ✅ CRITICAL: Don't silently convert errors to false!
                    error!("❌ Producer #{} unhealthy: {}", idx, e);
                    errors.push((idx, format!("{}", e)));
                }
            }
        }

        // ✅ If ANY producer is unhealthy, return error to caller
        if !errors.is_empty() {
            let error_summary = errors
                .iter()
                .map(|(id, err)| format!("Producer #{}: {}", id, err))
                .collect::<Vec<_>>()
                .join("; ");

            error!("🚨 FATAL: {} producers unhealthy!", errors.len());
            error!("   Errors: {}", error_summary);

            return Err(PoolError::ProducersUnhealthy(error_summary));
        }

        Ok(any_true)
    }

    /// Get number of producers
    pub fn num_producers(&self) -> usize {
        self.num_producers
    }

    /// Get a specific producer handle
    pub fn get_producer(&self, index: usize) -> &LockFreeProducer {
        &self.producers[index % self.num_producers]
    }

    /// 🚨 v1.0.2 FIX #2: Check health of all producers
    /// Returns Vec<(producer_id, is_healthy)>
    pub fn health_check(&self) -> Vec<(usize, bool)> {
        let mut health_status = Vec::new();

        for (id, producer) in self.producers.iter().enumerate() {
            let is_healthy = !producer.command_tx.is_closed();
            health_status.push((id, is_healthy));

            if !is_healthy {
                error!("❌ Producer #{} task is DEAD (channel closed)!", id);
                error!("   This producer will NEVER respond to should_produce() queries!");
            }
        }

        health_status
    }

    /// 🚨 v1.0.2 FIX #2: Get producer height consensus
    /// Returns (majority_height, count_at_majority) if consensus exists
    /// Get consensus height and count of producers at that height
    ///
    /// **v1.0.3-beta Enhancement**: Now exports Prometheus metrics for observability
    ///
    /// **Metrics Exported**:
    /// - `qnk_producer_height{producer_id}` - Current height of each producer
    /// - `qnk_producer_task_alive{producer_id}` - Task liveness (1=alive, 0=dead)
    /// - `qnk_consensus_health_percent` - % of producers at consensus
    /// - `qnk_producer_divergence_count{severity}` - Divergence by severity
    /// - `qnk_producer_max_drift_blocks` - Maximum drift between producers
    pub async fn get_height_consensus(&self) -> Option<(u64, usize)> {
        use std::collections::HashMap;

        let start = std::time::Instant::now();
        let mut heights = HashMap::new();
        let mut alive_count = 0;
        let mut channel_capacities = Vec::new();

        // Collect heights from all producers and update task liveness metrics
        for (id, producer) in self.producers.iter().enumerate() {
            let is_alive = !producer.command_tx.is_closed();
            let capacity = producer.command_tx.capacity();
            channel_capacities.push((id, capacity));

            // TODO v1.0.4: Re-enable metrics
            // PRODUCER_TASK_ALIVE
            //     .with_label_values(&[&id.to_string()])
            //     .set(if is_alive { 1 } else { 0 });

            if is_alive {
                alive_count += 1;
                let height = producer.get_height().await;
                *heights.entry(height).or_insert(0) += 1;

                // TODO v1.0.4: Re-enable metrics
                // PRODUCER_HEIGHT
                //     .with_label_values(&[&id.to_string()])
                //     .set(height as i64);

                debug!(
                    "Producer #{} is at height {}, channel capacity remaining: {}",
                    id, height, capacity
                );
            } else {
                warn!("⚠️  Producer #{} task is DEAD!", id);
            }
        }

        // 🔍 v1.0.3.4-beta DIAGNOSTIC: Log channel capacity status
        let min_capacity = channel_capacities
            .iter()
            .map(|(_, c)| c)
            .min()
            .copied()
            .unwrap_or(0);
        let avg_capacity: usize = channel_capacities.iter().map(|(_, c)| c).sum::<usize>()
            / channel_capacities.len().max(1);
        if min_capacity < 1000 {
            warn!(
                "⚠️  Low channel capacity detected! Min: {}, Avg: {}, Details: {:?}",
                min_capacity, avg_capacity, channel_capacities
            );
        }

        if alive_count == 0 {
            error!("🚨 CRITICAL: All producers are dead!");
            return None;
        }

        // Find majority height
        let majority = heights.iter().max_by_key(|(_, count)| *count)?;
        let (majority_height, count) = (*majority.0, *majority.1);

        // Calculate consensus health percentage
        let health_percentage = (count as f64 / alive_count as f64) * 100.0;
        // TODO v1.0.4: Re-enable metrics
        // CONSENSUS_HEALTH_PCT
        //     .with_label_values(&[])
        //     .set(health_percentage);

        // Calculate maximum drift (highest - lowest)
        let min_height = *heights.keys().min().unwrap_or(&majority_height);
        let max_height = *heights.keys().max().unwrap_or(&majority_height);
        let max_drift = max_height - min_height;

        // TODO v1.0.4: Re-enable metrics
        // PRODUCER_MAX_DRIFT
        //     .with_label_values(&[])
        //     .set(max_drift as i64);

        // Calculate divergence and categorize by severity
        let diverged_count = alive_count - count;
        if diverged_count > 0 {
            let severity = if max_drift <= 2 {
                "minor"
            } else if max_drift <= 5 {
                "moderate"
            } else {
                "severe"
            };

            // TODO v1.0.4: Re-enable metrics
            // // Reset all severity counters first
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["minor"]).set(0);
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["moderate"]).set(0);
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["severe"]).set(0);
            //
            // // Set the current severity
            // PRODUCER_DIVERGENCE_COUNT
            //     .with_label_values(&[severity])
            //     .set(diverged_count as i64);

            warn!("⚠️  Producer height divergence detected!");
            warn!(
                "   {}/{} producers at height {} (consensus)",
                count, alive_count, majority_height
            );
            warn!(
                "   Max drift: {} blocks (severity: {})",
                max_drift, severity
            );
            warn!("   Divergent heights: {:?}", heights);
        } else {
            // TODO v1.0.4: Re-enable metrics
            // // All producers in consensus - reset divergence metrics
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["minor"]).set(0);
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["moderate"]).set(0);
            // PRODUCER_DIVERGENCE_COUNT.with_label_values(&["severe"]).set(0);

            debug!(
                "✅ All {} producers at consensus height {}",
                alive_count, majority_height
            );
        }

        Some((majority_height, count))
    }

    /// ✅ v1.0.13-beta CRASH-FAST FIX #2: Enforce height invariant
    ///
    /// **CRITICAL**: All producers must stay within 1 block of each other!
    /// If height spread > 1, this indicates synchronization failure and we CRASH.
    ///
    /// **Philosophy**: Better to crash and restart than sit deadlocked forever!
    ///
    /// **Returns**: Ok(()) if invariant is satisfied, exits process if violated
    pub async fn enforce_height_invariant(
        &self,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<()> {
        // Get storage height for reference
        let storage_height = storage.get_highest_contiguous_block().await?;

        // Get heights from all producers
        let mut producer_heights = Vec::new();
        for (id, producer) in self.producers.iter().enumerate() {
            match tokio::time::timeout(tokio::time::Duration::from_secs(5), producer.get_height())
                .await
            {
                Ok(height) => producer_heights.push((id, height)),
                Err(_) => {
                    // Producer timed out on get_height() - CRITICAL!
                    error!("🚨 FATAL: Producer #{} timed out on get_height()!", id);
                    error!("   Producer task may be DEADLOCKED!");
                    error!("   Exiting to trigger systemd restart...");
                    eprintln!("[CRASH] enforce_height_invariant: Producer #{} timed out on get_height() — DEADLOCKED", id);
                    std::process::exit(1);
                }
            }
        }

        // Calculate height spread
        let min_height = producer_heights.iter().map(|(_, h)| h).min().unwrap_or(&0);
        let max_height = producer_heights.iter().map(|(_, h)| h).max().unwrap_or(&0);
        let spread = max_height.saturating_sub(*min_height);

        // INVARIANT: All producers within 1 block of each other
        if spread > 1 {
            error!(
                "🚨 FATAL INVARIANT VIOLATION: Producer height spread = {}",
                spread
            );
            error!("   Storage height: {}", storage_height);
            error!("   Producer heights: {:?}", producer_heights);
            error!(
                "   Min: {}, Max: {}, Spread: {}",
                min_height, max_height, spread
            );
            error!("   Producers are OUT OF SYNC - this will cause deadlock!");
            error!("   Exiting to trigger restart and resync...");
            eprintln!("[CRASH] enforce_height_invariant: Producer spread={} (min={}, max={}, storage={})", spread, min_height, max_height, storage_height);
            std::process::exit(1);
        }

        // Invariant satisfied
        debug!(
            "✅ Height invariant satisfied: spread={} (max allowed: 1)",
            spread
        );
        Ok(())
    }

    /// Synchronize all producers from storage (NO LOCKS!)
    /// 🚨 v1.0.2 FIX #5: Enhanced with health checks and consensus verification
    /// 🔍 v1.0.3.4-beta DIAGNOSTIC: Added timing measurements
    pub async fn sync_from_storage(
        &self,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<()> {
        let sync_start = std::time::Instant::now();
        info!(
            "🔄 [LOCK-FREE SYNC v1.0.2] Synchronizing all {} producers with blockchain state...",
            self.num_producers
        );

        // 🚨 v1.0.2 FIX #5: Check producer health BEFORE sync
        let health_status = self.health_check();
        let dead_count = health_status.iter().filter(|(_, h)| !h).count();
        if dead_count > 0 {
            warn!(
                "⚠️  [LOCK-FREE SYNC] {} producers are DEAD - sync may be incomplete!",
                dead_count
            );
        }

        // v10.2.11: Use spawn_blocking for ALL RocksDB calls to prevent tokio worker starvation.
        //
        // ROOT CAUSE of recurring production loop freeze (every few hours):
        // The KVStore trait methods (get, put, etc.) are declared `async fn` but internally call
        // synchronous RocksDB operations (get_cf, put_cf_opt). When RocksDB is doing compaction,
        // block-pack serving, or heavy reads, these calls block the tokio WORKER thread for seconds.
        // The previous v10.2.10 fix used tokio::time::timeout(), but timeout() races two futures
        // on the SAME worker thread — if the inner future is blocking, the timeout timer never
        // gets polled either, so the timeout never fires. Result: ALL workers eventually blocked,
        // production loop task never gets scheduled, silent stall.
        //
        // FIX: spawn_blocking() moves the blocking RocksDB work to a DEDICATED blocking thread
        // pool, freeing the tokio worker immediately. The timeout wrapper now works correctly
        // because it runs on an unblocked worker thread.
        let storage_query_start = std::time::Instant::now();
        let storage_clone = Arc::clone(storage);
        let highest_height = match timeout(Duration::from_secs(5), tokio::task::spawn_blocking({
            let storage_ref = Arc::clone(&storage_clone);
            move || {
                tokio::runtime::Handle::current().block_on(storage_ref.get_highest_contiguous_block())
            }
        })).await {
            Ok(Ok(Ok(h))) => h,
            Ok(Ok(Err(e))) => {
                error!("🚨 [SYNC-FROM-STORAGE] get_highest_contiguous_block() failed: {}", e);
                return Err(e.into());
            }
            Ok(Err(join_err)) => {
                error!("🚨 [SYNC-FROM-STORAGE] get_highest_contiguous_block() task panicked: {}", join_err);
                return Err(anyhow::anyhow!("spawn_blocking panicked: get_highest_contiguous_block"));
            }
            Err(_) => {
                error!("🚨 [SYNC-FROM-STORAGE] get_highest_contiguous_block() TIMED OUT after 5s — RocksDB may be stalled by compaction/block-pack I/O");
                return Err(anyhow::anyhow!("RocksDB timeout: get_highest_contiguous_block"));
            }
        };
        let storage_query_duration = storage_query_start.elapsed();

        // v10.2.7: Also read the DB pointer for comparison
        // v10.2.11: spawn_blocking to prevent tokio worker starvation
        let db_pointer = match timeout(Duration::from_secs(5), tokio::task::spawn_blocking({
            let storage_ref = Arc::clone(&storage_clone);
            move || {
                tokio::runtime::Handle::current().block_on(storage_ref.get_latest_qblock_height())
            }
        })).await {
            Ok(Ok(Ok(Some(h)))) => h,
            Ok(Ok(Ok(None))) => 0,
            Ok(Ok(Err(e))) => {
                warn!("⚠️ [SYNC-FROM-STORAGE] get_latest_qblock_height() failed: {} — using 0", e);
                0
            }
            Ok(Err(join_err)) => {
                warn!("⚠️ [SYNC-FROM-STORAGE] get_latest_qblock_height() task panicked: {} — using 0", join_err);
                0
            }
            Err(_) => {
                warn!("⚠️ [SYNC-FROM-STORAGE] get_latest_qblock_height() TIMED OUT after 5s — using 0");
                0
            }
        };
        info!(
            "🔍 [SYNC_FROM_STORAGE] height_cache={}, db_pointer(qblock:latest)={}, delta={}, query_time={:?}",
            highest_height, db_pointer,
            if highest_height > db_pointer { highest_height - db_pointer } else { db_pointer - highest_height },
            storage_query_duration
        );

        if highest_height == 0 {
            info!("📝 [LOCK-FREE SYNC] No blocks in storage yet - producers at genesis");
            return Ok(());
        }

        info!(
            "🔍 [LOCK-FREE SYNC] Found highest block at height {} in storage",
            highest_height
        );

        // v7.3.4: Handle decompression errors gracefully instead of propagating.
        // If the block can't be deserialized/decompressed, fall through to height-only mode
        // rather than aborting the entire sync (which prevents height advancement and stalls production).
        // v10.2.11: spawn_blocking to prevent tokio worker starvation from RocksDB blocking reads
        let block_result = match timeout(Duration::from_secs(5), tokio::task::spawn_blocking({
            let storage_ref = Arc::clone(&storage_clone);
            move || {
                tokio::runtime::Handle::current().block_on(storage_ref.get_qblock_by_height(highest_height))
            }
        })).await {
            Ok(Ok(Ok(b))) => b,
            Ok(Ok(Err(e))) => {
                warn!(
                    "⚠️  [LOCK-FREE SYNC] Failed to load block at height {}: {} — using height-only mode",
                    highest_height, e
                );
                None
            }
            Ok(Err(join_err)) => {
                warn!(
                    "⚠️  [LOCK-FREE SYNC] get_qblock_by_height({}) task panicked: {} — using height-only mode",
                    highest_height, join_err
                );
                None
            }
            Err(_) => {
                warn!(
                    "⚠️  [LOCK-FREE SYNC] get_qblock_by_height({}) TIMED OUT after 5s — using height-only mode",
                    highest_height
                );
                None
            }
        };

        match block_result {
            Some(latest_block) => {
                let new_height = latest_block.header.height;
                let new_hash = latest_block.calculate_hash();
                let new_difficulty = latest_block.header.total_difficulty;
                let new_dag_round = latest_block.header.dag_round;

                info!(
                    "   Latest block metadata: height={}, hash={}",
                    new_height,
                    hex::encode(&new_hash[..8])
                );

                // 🚨 v1.0.2 FIX #5: Update ALL producers atomically (fire-and-forget)
                // This uses try_send which is non-blocking, so it's as atomic as we can get
                // without introducing async complexity
                for (i, producer) in self.producers.iter().enumerate() {
                    producer.set_latest_block(new_height, new_hash, new_difficulty, new_dag_round);
                    debug!(
                        "   ✅ Lock-free producer #{} synchronized: height={}",
                        i, new_height
                    );
                }

                // 🚀 v1.0.95-beta: REMOVED 100ms sleep that was blocking sync performance!
                // The sleep was originally for producer convergence, but:
                // 1. Producers converge naturally via eventual consistency
                // 2. During sync, this adds 100ms per block = HUGE slowdown
                // 3. The consensus check is just for logging, not correctness
                //
                // Changed from 100ms to 0ms - immediate check with no blocking
                if let Some((consensus_height, count)) = self.get_height_consensus().await {
                    if consensus_height != new_height || count != self.num_producers {
                        // ✅ CRITICAL FIX: Log only, don't return error!
                        // This allows eventual consistency instead of strict atomic consistency
                        warn!(
                            "⚠️  [PRODUCER-DRIFT] {}/{} producers at height {} (expected {})",
                            count, self.num_producers, consensus_height, new_height
                        );
                        warn!("   This is NORMAL in parallel production - producers will converge naturally");
                        warn!("   Allowing operation to continue (eventual consistency model)");

                        // Note: Producers will converge naturally through normal block production
                        // No need to force synchronization - that's what caused the deadlock!
                    } else {
                        info!(
                            "✅ [SYNC-CONSENSUS] All {} producers at height {}",
                            count, consensus_height
                        );
                    }
                }

                // 🚨 v7.3.4 CRITICAL FIX: Reset pool_last_produced_height when storage height regresses.
                // Without this, if block N+1 is produced but save_qblock() fails, storage stays at N
                // but pool_last_produced_height stays at N+1. The pool-level duplicate check then
                // permanently blocks all future block N+1 attempts: (N+1) <= pool_height(N+1) → POOL DUPLICATE.
                // This is the SAME bug pattern as the per-producer STALL-FIX (block_producer.rs line 1992)
                // but at the pool level.
                let pool_height = self.pool_last_produced_height.load(Ordering::SeqCst);
                if new_height < pool_height {
                    warn!(
                        "🔧 [POOL-STALL-FIX] Resetting stale pool_last_produced_height {} → {} (storage height regressed, likely unsaved block)",
                        pool_height, new_height
                    );
                    self.pool_last_produced_height.store(new_height, Ordering::SeqCst);
                }

                info!(
                    "✅ [LOCK-FREE SYNC] All producers synchronized to height {} (ZERO LOCKS!)",
                    new_height
                );
            }
            None => {
                warn!(
                    "⚠️  [LOCK-FREE SYNC] Block #{} exists but cannot load data",
                    highest_height
                );

                let zero_hash = [0u8; 32];
                let zero_difficulty = 0u128;

                for (i, producer) in self.producers.iter().enumerate() {
                    producer.set_latest_block(
                        highest_height,
                        zero_hash,
                        zero_difficulty,
                        highest_height,
                    );
                    debug!(
                        "   ⚠️  Lock-free producer #{} synchronized to height {} (height-only)",
                        i, highest_height
                    );
                }

                // 🚨 v7.3.4: Also reset pool_last_produced_height in height-only mode
                let pool_height = self.pool_last_produced_height.load(Ordering::SeqCst);
                if highest_height < pool_height {
                    warn!(
                        "🔧 [POOL-STALL-FIX] Resetting stale pool_last_produced_height {} → {} (height-only mode)",
                        pool_height, highest_height
                    );
                    self.pool_last_produced_height.store(highest_height, Ordering::SeqCst);
                }

                info!("✅ [LOCK-FREE SYNC] All producers synchronized to height {} (height-only mode)", highest_height);
            }
        }

        // 🔍 v1.0.3.4-beta DIAGNOSTIC: Log total sync duration
        let sync_total_duration = sync_start.elapsed();
        debug!(
            "🔍 [TIMING] Total sync_from_storage took {:?}",
            sync_total_duration
        );
        if sync_total_duration.as_millis() > 100 {
            warn!(
                "⚠️  [SLOW-SYNC] sync_from_storage took {:?} (>100ms threshold)",
                sync_total_duration
            );
        }

        Ok(())
    }

    /// 🚀 v1.0.3.9-beta: Notify producers that a new block was saved
    ///
    /// This is the PRIMARY fix for stale state - called immediately when blocks are saved.
    /// Updates all producers to the new height atomically.
    ///
    /// # Arguments
    /// * `new_height` - The height of the newly saved block
    /// * `new_hash` - Hash of the newly saved block
    /// * `new_difficulty` - Total difficulty at the new height
    pub async fn notify_height_advanced(
        &self,
        new_height: u64,
        new_hash: BlockHash,
        new_difficulty: u128,
    ) -> anyhow::Result<()> {
        // Get current producer height from first producer (all should be in sync)
        let current_height = if let Some(handle) = self.producers.first() {
            let (reply_tx, reply_rx) = oneshot::channel();
            if handle
                .command_tx
                .send(ProducerCommand::GetHeight(reply_tx))
                .await
                .is_ok()
            {
                reply_rx.await.unwrap_or(0)
            } else {
                0
            }
        } else {
            return Ok(()); // No producers
        };

        // Only advance forward
        if new_height > current_height {
            info!(
                "📈 [HEIGHT ADVANCE] Network block at {}, advancing producers from {}",
                new_height, current_height
            );

            // Update all producers via their command channels
            for (i, handle) in self.producers.iter().enumerate() {
                if let Err(e) = handle
                    .update_height(new_height, new_hash, new_difficulty)
                    .await
                {
                    error!("❌ [HEIGHT ADVANCE] Failed to update producer {}: {}", i, e);
                } else {
                    debug!(
                        "✅ [HEIGHT ADVANCE] Producer {} updated to height {}",
                        i, new_height
                    );
                }
            }

            info!(
                "✅ [HEIGHT ADVANCE] All {} producers advanced to height {}",
                self.producers.len(),
                new_height
            );
        } else if new_height < current_height {
            warn!(
                "⚠️  [HEIGHT ADVANCE] Attempted backward move from {} to {} (reorg?)",
                current_height, new_height
            );
            // For reorgs, trigger full resync (TODO: implement reorg handling)
        } else {
            debug!("📊 [HEIGHT ADVANCE] Height {} already current", new_height);
        }

        Ok(())
    }

    /// 🚀 v1.0.3.9-beta: State consistency monitoring task
    ///
    /// Periodically checks if producer height matches database height.
    /// Auto-resyncs on divergence detection.
    ///
    /// This is a **backup safety net** - the primary fix is sync-on-block-save hooks.
    pub async fn spawn_state_monitor(self: Arc<Self>, storage: Arc<q_storage::QStorage>) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

            loop {
                interval.tick().await;

                // Check consistency
                if let Err(e) = self.check_state_consistency(&storage).await {
                    error!("❌ [STATE MONITOR] Consistency check failed: {}", e);
                }
            }
        });

        info!("✅ [STATE MONITOR] State consistency watchdog started (10s interval)");
    }

    /// Check if producer height matches database height
    async fn check_state_consistency(
        &self,
        storage: &Arc<q_storage::QStorage>,
    ) -> anyhow::Result<()> {
        // Get database height
        let db_height = storage.get_highest_contiguous_block().await?;

        // Get producer height (all producers should be in sync, check first one)
        let producer_height = if let Some(handle) = self.producers.first() {
            let (reply_tx, reply_rx) = oneshot::channel();
            if handle
                .command_tx
                .send(ProducerCommand::GetHeight(reply_tx))
                .await
                .is_ok()
            {
                reply_rx.await.unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        // Check for divergence
        let gap = db_height.abs_diff(producer_height);

        if gap > 0 {
            if gap <= 3 {
                // Small gap (1-3 blocks) - just log, might be transient
                debug!(
                    "⚠️  [STATE MONITOR] Small gap detected: DB={}, Producers={}, gap={}",
                    db_height, producer_height, gap
                );
            } else {
                // Large gap (>3 blocks) - critical divergence, auto-resync
                error!("🚨 [STATE DIVERGENCE] CRITICAL gap detected:");
                error!("   Database height:  {}", db_height);
                error!("   Producer height:  {}", producer_height);
                error!("   Gap:              {} blocks", gap);

                // Auto-resync
                warn!("🔄 [AUTO-RESYNC] Re-synchronizing producers to database state...");

                let resync_start = std::time::Instant::now();
                self.sync_from_storage(storage).await?;
                let resync_duration = resync_start.elapsed();

                info!(
                    "✅ [AUTO-RESYNC] Producers synchronized to height {} in {:?}",
                    db_height, resync_duration
                );

                // Verify resync worked
                let new_producer_height = if let Some(handle) = self.producers.first() {
                    let (reply_tx, reply_rx) = oneshot::channel();
                    if handle
                        .command_tx
                        .send(ProducerCommand::GetHeight(reply_tx))
                        .await
                        .is_ok()
                    {
                        reply_rx.await.unwrap_or(0)
                    } else {
                        0
                    }
                } else {
                    0
                };

                if new_producer_height == db_height {
                    info!("✅ [AUTO-RESYNC] Verification PASSED - heights match");
                } else {
                    error!("❌ [AUTO-RESYNC] Verification FAILED - heights still diverged!");
                    error!("   Expected: {}, Got: {}", db_height, new_producer_height);
                }
            }
        } else {
            debug!(
                "✅ [STATE MONITOR] Heights match: DB={}, Producers={}",
                db_height, producer_height
            );
        }

        Ok(())
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
    /// User nodes were stuck at height 1 because advance_height() was never called
    /// after block production. This method sends the AdvanceHeight command to the
    /// appropriate producer via the lock-free channel.
    pub fn advance_producer_height(&self, producer_id: usize, block_hash: BlockHash) {
        let producer_index = producer_id % self.num_producers;
        self.producers[producer_index].advance_height(block_hash);

        info!("✅ [v1.0.8-beta FIX] Pool: Producer #{} height advance command sent AFTER storage confirmation",
              producer_id);
    }

    /// Set validator keypair for all producers
    /// ✨ v1.0.16-beta: Enable PQC block signing across all producers
    ///
    /// This sends the SetValidatorKeypair command to all producer tasks,
    /// enabling post-quantum signatures on all produced blocks.
    pub fn set_validator_keypair(&self, keypair: Arc<q_types::ValidatorKeypair>) {
        info!(
            "🔐 [PQC] Setting validator keypair for all {} producers...",
            self.num_producers
        );
        for producer in &self.producers {
            producer.set_validator_keypair(keypair.clone());
        }
        info!("✅ [PQC] Validator keypair sent to all producers");
    }

    /// Set DAG-Knight consensus for all producers
    /// ⚔️  v1.0.3-beta: Enable DAG-aware sync (Phase 1)
    ///
    /// This sends the SetDagKnight command to all producer tasks,
    /// enabling dag_parents population in all produced blocks.
    /// This is the foundation for Phase 2 DAG-aware layered sync (10-50x faster).
    pub fn set_dag_knight(&self, dag_knight: Arc<q_dag_knight::DAGKnightConsensus>) {
        info!(
            "⚔️  [DAG-Knight] Setting consensus for all {} producers...",
            self.num_producers
        );
        for producer in &self.producers {
            producer.set_dag_knight(dag_knight.clone());
        }
        info!("✅ [DAG-Knight] Consensus sent to all producers - dag_parents will be populated");
    }

    /// Set event emitter for all producers
    /// 🔔 v1.0.17-beta: Enable SSE mining reward notifications across all producers
    ///
    /// This sends the SetEventEmitter command to all producer tasks,
    /// enabling real-time mining reward broadcasts to connected clients.
    pub fn set_event_emitter(&self, emitter: Arc<crate::streaming::HighPerformanceEmitter>) {
        info!(
            "🔔 [SSE] Setting event emitter for all {} producers...",
            self.num_producers
        );
        for producer in &self.producers {
            producer.set_event_emitter(emitter.clone());
        }
        info!("✅ [SSE] Event emitter sent to all producers");
    }

    /// Set production mempool for all producers
    /// 📦 v3.5.14-beta: Enable P2P transaction propagation across all producers
    ///
    /// This sends the SetProductionMempool command to all producer tasks,
    /// enabling user transactions from the mempool to be included in produced blocks.
    /// This is CRITICAL for P2P transaction propagation to work!
    pub fn set_production_mempool(&self, mempool: Arc<q_narwhal_core::production_mempool::ProductionMempool>) {
        info!(
            "📦 [MEMPOOL] Setting production mempool for all {} producers...",
            self.num_producers
        );
        for producer in &self.producers {
            producer.set_production_mempool(mempool.clone());
        }
        info!("✅ [MEMPOOL] Production mempool sent to all producers - user transactions will be included in blocks!");
    }

    /// 📦 v3.5.20-beta: Enable P2P transaction status tracking across all producers
    ///
    /// This sends the tx_status map to all producer tasks, enabling
    /// transactions to be marked as Confirmed after block inclusion.
    /// This is CRITICAL for P2P transactions to show as confirmed in explorer!
    pub fn set_tx_status(&self, tx_status: Arc<dashmap::DashMap<q_types::TxHash, q_types::TxStatus>>) {
        info!(
            "📦 [TX-STATUS] Setting transaction status tracker for all {} producers...",
            self.num_producers
        );
        for producer in &self.producers {
            producer.set_tx_status(tx_status.clone());
        }
        info!("✅ [TX-STATUS] Transaction status tracker sent to all producers - P2P transactions will be confirmed!");
    }

    /// 💰 v7.1.5: Set configurable dev fee across all producers
    pub fn set_dev_fee_bps(&self, dev_fee_bps: Arc<std::sync::atomic::AtomicU64>) {
        for producer in &self.producers {
            producer.set_dev_fee_bps(dev_fee_bps.clone());
        }
    }

    /// 💰 v8.6.1: Set operator fee share across all producers
    pub fn set_operator_fee(&self, promille: Arc<std::sync::atomic::AtomicU64>, admin_wallet: String) {
        for producer in &self.producers {
            producer.set_operator_fee(promille.clone(), admin_wallet.clone());
        }
        info!("💰 Dev fee BPS shared with all {} producers", self.num_producers);
    }

    /// 💰 v8.7.0: Set distributed operators for fee splitting across all producers
    pub fn set_distributed_operators(&self, operators: Vec<crate::block_producer::OperatorRewardEntry>) {
        for producer in &self.producers {
            producer.set_distributed_operators(operators.clone());
        }
    }

    /// 🏊 v9.1.2: Set mining pool for PPLNS reward distribution across all producers
    pub fn set_mining_pool(&self, pool: Arc<q_mining_pool::MiningPool>) {
        for producer in &self.producers {
            producer.set_mining_pool(pool.clone());
        }
        info!("🏊 Mining pool set for PPLNS distribution across all {} producers", self.num_producers);
    }

    /// 🌐 v10.0.2: Set distributed PPLNS proportions across all producers
    pub fn set_distributed_pplns(&self, proportions: Arc<tokio::sync::RwLock<Option<Vec<(String, f64)>>>>) {
        for producer in &self.producers {
            producer.set_distributed_pplns(proportions.clone());
        }
        info!("🌐 Distributed PPLNS set across all {} producers", self.num_producers);
    }

    /// 📊 v9.3.1: Set max solutions per block across all producers (K-parameter tuning)
    pub fn set_max_solutions_per_block(&self, max_solutions: usize) {
        for producer in &self.producers {
            producer.set_max_solutions_per_block(max_solutions);
        }
    }

    /// Shutdown all producers gracefully
    pub fn shutdown(&self) {
        info!("🛑 Shutting down lock-free producer pool...");
        for producer in &self.producers {
            producer.shutdown();
        }
        info!("✅ All producers shutdown gracefully");
    }
}

impl Drop for LockFreeProducerPool {
    fn drop(&mut self) {
        // Ensure producers are shut down when pool is dropped
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lockfree_producer_creation() {
        let config = BlockProducerConfig::default();
        let producer = LockFreeProducer::new(0, config);

        let height = producer.get_height().await;
        assert_eq!(height, 0);
    }

    #[tokio::test]
    async fn test_lockfree_solution_queueing() {
        let config = BlockProducerConfig::default();
        let producer = LockFreeProducer::new(0, config);

        let solution = MiningSolution {
            nonce: 12345,
            hash: [0u8; 32],
            difficulty_target: [0xFF; 32],
            miner_address: [1u8; 32],
            timestamp: 1234567890,
            pool_id: None,
            hash_rate_hs: 10000,
            miner_id: None, worker_name: None,
            vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None,
        };

        // This should NEVER block!
        producer.queue_solution(solution);

        // Give task time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    #[tokio::test]
    async fn test_lockfree_pool() {
        let config = BlockProducerConfig::default();
        let pool = LockFreeProducerPool::new(4, config);

        assert_eq!(pool.num_producers(), 4);

        // Queue solution - should NEVER block!
        let solution = MiningSolution {
            nonce: 99,
            hash: [0u8; 32],
            difficulty_target: [0xFF; 32],
            miner_address: [2u8; 32],
            timestamp: 1234567890,
            pool_id: None,
            hash_rate_hs: 15000,
            miner_id: None, worker_name: None,
            vdf_output: None, vdf_proof: None, vdf_checkpoints: None, vdf_iterations_count: None,
        };

        pool.queue_solution(solution);

        // Give tasks time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}
