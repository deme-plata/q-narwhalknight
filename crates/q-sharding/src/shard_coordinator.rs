// Q-NarwhalKnight Shard Coordinator
// Central orchestration and management of shard operations

use crate::cross_shard_bridge::CrossShardBridge;
use crate::load_balancer::LoadBalancer;
use crate::{ShardConfig, ShardId, ShardMetrics, ShardingEngine};
use anyhow::Result;
use futures::future::join_all;
use q_types::{Hash256, NodeId, Transaction};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing;

// Utility function to generate random Hash256
fn generate_random_hash256() -> Hash256 {
    let mut hasher = Sha3_256::new();
    hasher.update(&uuid::Uuid::new_v4().to_string().as_bytes());
    let hash = hasher.finalize();
    let mut array = [0u8; 32];
    array.copy_from_slice(&hash[..32]);
    array
}

/// Central coordinator for shard management and orchestration
#[derive(Debug)]
pub struct ShardCoordinator {
    coordinator_id: String,
    config: ShardConfig,
    active_shards: Arc<RwLock<HashMap<u32, ShardInfo>>>,
    rebalancing_state: Arc<Mutex<RebalancingState>>,
    operation_queue: Arc<RwLock<VecDeque<CoordinatorOperation>>>,
    metrics_collector: Arc<RwLock<CoordinatorMetrics>>,
    cross_shard_bridge: Option<Arc<CrossShardBridge>>,
    load_balancer: Option<Arc<LoadBalancer>>,
}

/// Information about an active shard
#[derive(Debug, Clone)]
struct ShardInfo {
    shard_id: u32,
    shard_type: ShardType,
    status: ShardStatus,
    last_heartbeat: Instant,
    performance_metrics: Option<ShardMetrics>,
    migration_state: Option<MigrationState>,
}

/// Type of shard
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum ShardType {
    Consensus,
    State,
}

/// Status of a shard
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum ShardStatus {
    Active,
    Migrating,
    Splitting,
    Merging,
    Inactive,
    Failed,
}

/// Migration state for shard rebalancing
#[derive(Debug, Clone)]
struct MigrationState {
    migration_id: Hash256,
    source_shard: u32,
    target_shard: u32,
    migration_type: MigrationType,
    progress: f64,
    started_at: Instant,
    estimated_completion: Option<Instant>,
}

/// Types of migration operations
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum MigrationType {
    LoadBalance, // Move work from overloaded to underloaded shard
    Split,       // Split one shard into two
    Merge,       // Merge two shards into one
    Repair,      // Move work away from failed shard
}

/// Current system rebalancing state
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RebalancingState {
    is_rebalancing: bool,
    rebalance_id: Hash256,
    started_at: Option<Instant>,
    operations_in_progress: HashSet<Hash256>,
    total_operations: usize,
    completed_operations: usize,
}

/// Operations managed by the coordinator
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum CoordinatorOperation {
    RebalanceShards {
        operation_id: Hash256,
        trigger_reason: RebalanceTrigger,
        target_shards: Vec<u32>,
    },
    SplitShard {
        operation_id: Hash256,
        source_shard: u32,
        split_criterion: SplitCriterion,
    },
    MergeShards {
        operation_id: Hash256,
        source_shards: Vec<u32>,
        target_shard: u32,
    },
    RecoverFromFailure {
        operation_id: Hash256,
        failed_shard: u32,
        recovery_strategy: RecoveryStrategy,
    },
}

/// Reasons for triggering rebalancing
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum RebalanceTrigger {
    LoadImbalance,
    CapacityThreshold,
    PerformanceDegradation,
    FailureRecovery,
    Scheduled,
}

/// Criteria for splitting a shard
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum SplitCriterion {
    LoadThreshold,
    StateSize,
    GeographicDistribution,
}

/// Recovery strategies for failed shards
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum RecoveryStrategy {
    Redistribute,
    Replicate,
    Checkpoint,
}

/// Coordinator performance metrics
#[derive(Debug, Clone, Default)]
struct CoordinatorMetrics {
    total_operations: u64,
    successful_operations: u64,
    failed_operations: u64,
    average_operation_time_ms: f64,
    active_migrations: usize,
    total_rebalances: u64,
    last_rebalance_time_ms: u64,
}

impl Default for RebalancingState {
    fn default() -> Self {
        Self {
            is_rebalancing: false,
            rebalance_id: Hash256::default(),
            started_at: None,
            operations_in_progress: HashSet::new(),
            total_operations: 0,
            completed_operations: 0,
        }
    }
}

impl ShardCoordinator {
    /// Create new shard coordinator
    pub async fn new(config: ShardConfig) -> Result<Self> {
        let coordinator_id = format!("coordinator_{}", rand::random::<u64>());

        tracing::info!("Creating shard coordinator: {}", coordinator_id);

        let active_shards = Arc::new(RwLock::new(HashMap::new()));

        // Initialize shard info for all configured shards
        {
            let mut shards = active_shards.write().await;

            // Add consensus shards
            for shard_id in 0..config.consensus_shards {
                shards.insert(
                    shard_id,
                    ShardInfo {
                        shard_id,
                        shard_type: ShardType::Consensus,
                        status: ShardStatus::Active,
                        last_heartbeat: Instant::now(),
                        performance_metrics: None,
                        migration_state: None,
                    },
                );
            }
        }

        Ok(Self {
            coordinator_id,
            config,
            active_shards,
            rebalancing_state: Arc::new(Mutex::new(RebalancingState::default())),
            operation_queue: Arc::new(RwLock::new(VecDeque::new())),
            metrics_collector: Arc::new(RwLock::new(CoordinatorMetrics::default())),
            cross_shard_bridge: None,
            load_balancer: None,
        })
    }

    /// Set cross-shard bridge for coordination
    pub fn set_cross_shard_bridge(&mut self, bridge: Arc<CrossShardBridge>) {
        self.cross_shard_bridge = Some(bridge);
    }

    /// Set load balancer for rebalancing decisions
    pub fn set_load_balancer(&mut self, balancer: Arc<LoadBalancer>) {
        self.load_balancer = Some(balancer);
    }

    /// Update shard metrics and check for rebalancing needs
    pub async fn update_shard_metrics(&self, shard_id: u32, metrics: ShardMetrics) -> Result<()> {
        let mut shards = self.active_shards.write().await;

        if let Some(shard_info) = shards.get_mut(&shard_id) {
            shard_info.performance_metrics = Some(metrics);
            shard_info.last_heartbeat = Instant::now();

            // Check if shard needs attention
            self.assess_shard_health(shard_info).await?;
        }

        Ok(())
    }

    /// Assess shard health and queue operations if needed
    async fn assess_shard_health(&self, shard_info: &mut ShardInfo) -> Result<()> {
        if let Some(ref metrics) = shard_info.performance_metrics {
            // Check for overload conditions
            if metrics.cpu_utilization > 0.9 || metrics.queue_depth > 5000 {
                self.queue_rebalance_operation(
                    shard_info.shard_id,
                    RebalanceTrigger::LoadImbalance,
                )
                .await?;
            }

            // Check for failure conditions
            if metrics.cpu_utilization == 0.0 && metrics.transactions_per_second == 0.0 {
                shard_info.status = ShardStatus::Failed;
                self.queue_recovery_operation(shard_info.shard_id).await?;
            }
        }

        // Check for stale heartbeat
        if shard_info.last_heartbeat.elapsed() > Duration::from_secs(120) {
            shard_info.status = ShardStatus::Failed;
            self.queue_recovery_operation(shard_info.shard_id).await?;
        }

        Ok(())
    }

    /// Queue rebalancing operation
    async fn queue_rebalance_operation(
        &self,
        shard_id: u32,
        trigger: RebalanceTrigger,
    ) -> Result<()> {
        let operation_id = generate_random_hash256();
        let operation = CoordinatorOperation::RebalanceShards {
            operation_id,
            trigger_reason: trigger,
            target_shards: vec![shard_id],
        };

        let mut queue = self.operation_queue.write().await;
        queue.push_back(operation);

        tracing::info!(
            "Queued rebalancing operation {:?} for shard {}",
            operation_id,
            shard_id
        );

        Ok(())
    }

    /// Queue recovery operation for failed shard
    async fn queue_recovery_operation(&self, failed_shard: u32) -> Result<()> {
        let operation_id = generate_random_hash256();
        let operation = CoordinatorOperation::RecoverFromFailure {
            operation_id,
            failed_shard,
            recovery_strategy: RecoveryStrategy::Redistribute,
        };

        let mut queue = self.operation_queue.write().await;
        queue.push_back(operation);

        tracing::warn!(
            "Queued recovery operation {:?} for failed shard {}",
            operation_id,
            failed_shard
        );

        Ok(())
    }

    /// Process queued operations
    pub async fn process_operations(&self) -> Result<usize> {
        let mut processed_count = 0;

        // Check if we're already rebalancing
        {
            let rebalance_state = self.rebalancing_state.lock().await;
            if rebalance_state.is_rebalancing {
                tracing::debug!("Rebalancing already in progress, skipping operation processing");
                return Ok(0);
            }
        }

        // Process operations from queue
        let operations = {
            let mut queue = self.operation_queue.write().await;
            let mut ops = Vec::new();

            // Take up to 5 operations to process in batch
            for _ in 0..5 {
                if let Some(op) = queue.pop_front() {
                    ops.push(op);
                } else {
                    break;
                }
            }

            ops
        };

        for operation in operations {
            match self.execute_operation(operation).await {
                Ok(_) => {
                    processed_count += 1;

                    let mut metrics = self.metrics_collector.write().await;
                    metrics.successful_operations += 1;
                }
                Err(e) => {
                    tracing::error!("Failed to execute coordinator operation: {}", e);

                    let mut metrics = self.metrics_collector.write().await;
                    metrics.failed_operations += 1;
                }
            }
        }

        if processed_count > 0 {
            tracing::info!("Processed {} coordinator operations", processed_count);
        }

        Ok(processed_count)
    }

    /// Execute a specific coordinator operation
    async fn execute_operation(&self, operation: CoordinatorOperation) -> Result<()> {
        let start_time = Instant::now();

        match operation {
            CoordinatorOperation::RebalanceShards {
                operation_id,
                trigger_reason,
                target_shards,
            } => {
                self.execute_rebalance_operation(operation_id, trigger_reason, target_shards)
                    .await?;
            }

            CoordinatorOperation::SplitShard {
                operation_id,
                source_shard,
                split_criterion,
            } => {
                self.execute_split_operation(operation_id, source_shard, split_criterion)
                    .await?;
            }

            CoordinatorOperation::MergeShards {
                operation_id,
                source_shards,
                target_shard,
            } => {
                self.execute_merge_operation(operation_id, source_shards, target_shard)
                    .await?;
            }

            CoordinatorOperation::RecoverFromFailure {
                operation_id,
                failed_shard,
                recovery_strategy,
            } => {
                self.execute_recovery_operation(operation_id, failed_shard, recovery_strategy)
                    .await?;
            }
        }

        // Update metrics
        let execution_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics_collector.write().await;
        metrics.total_operations += 1;
        metrics.average_operation_time_ms = (metrics.average_operation_time_ms
            * (metrics.total_operations - 1) as f64
            + execution_time)
            / metrics.total_operations as f64;

        Ok(())
    }

    /// Execute rebalancing operation
    async fn execute_rebalance_operation(
        &self,
        operation_id: Hash256,
        _trigger: RebalanceTrigger,
        target_shards: Vec<u32>,
    ) -> Result<()> {
        tracing::info!(
            "Executing rebalancing operation {:?} for {} shards",
            operation_id,
            target_shards.len()
        );

        // Set rebalancing state
        {
            let mut rebalance_state = self.rebalancing_state.lock().await;
            rebalance_state.is_rebalancing = true;
            rebalance_state.rebalance_id = operation_id;
            rebalance_state.started_at = Some(Instant::now());
            rebalance_state.total_operations = target_shards.len();
            rebalance_state.completed_operations = 0;
        }

        // Use load balancer to determine migration targets
        if let Some(ref load_balancer) = self.load_balancer {
            for &source_shard in &target_shards {
                if let Ok(Some(target_shard)) =
                    load_balancer.get_migration_target(source_shard).await
                {
                    tracing::info!(
                        "Migrating load from shard {} to shard {}",
                        source_shard,
                        target_shard
                    );

                    // Create migration state
                    let migration_id = {
                        use sha3::{Digest, Sha3_256};
                        let mut hasher = Sha3_256::new();
                        hasher.update(&uuid::Uuid::new_v4().to_string().as_bytes());
                        let hash = hasher.finalize();
                        let mut array = [0u8; 32];
                        array.copy_from_slice(&hash[..32]);
                        array
                    };
                    let migration_state = MigrationState {
                        migration_id,
                        source_shard,
                        target_shard,
                        migration_type: MigrationType::LoadBalance,
                        progress: 0.0,
                        started_at: Instant::now(),
                        estimated_completion: Some(Instant::now() + Duration::from_secs(60)),
                    };

                    // Update shard info
                    let mut shards = self.active_shards.write().await;
                    if let Some(shard_info) = shards.get_mut(&source_shard) {
                        shard_info.status = ShardStatus::Migrating;
                        shard_info.migration_state = Some(migration_state);
                    }

                    // Simulate migration process (in real implementation, this would coordinate actual data movement)
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    // Complete migration
                    if let Some(shard_info) = shards.get_mut(&source_shard) {
                        shard_info.status = ShardStatus::Active;
                        shard_info.migration_state = None;
                    }
                }
            }
        }

        // Complete rebalancing
        {
            let mut rebalance_state = self.rebalancing_state.lock().await;
            rebalance_state.is_rebalancing = false;
            rebalance_state.completed_operations = target_shards.len();

            let mut metrics = self.metrics_collector.write().await;
            metrics.total_rebalances += 1;
            if let Some(started_at) = rebalance_state.started_at {
                metrics.last_rebalance_time_ms = started_at.elapsed().as_millis() as u64;
            }
        }

        tracing::info!("Completed rebalancing operation {:?}", operation_id);

        Ok(())
    }

    /// Execute shard split operation
    async fn execute_split_operation(
        &self,
        operation_id: Hash256,
        source_shard: u32,
        _criterion: SplitCriterion,
    ) -> Result<()> {
        tracing::info!(
            "Executing shard split operation {:?} for shard {}",
            operation_id,
            source_shard
        );

        // Find next available shard ID
        let new_shard_id = {
            let shards = self.active_shards.read().await;
            let mut max_id = 0;
            for &shard_id in shards.keys() {
                max_id = max_id.max(shard_id);
            }
            max_id + 1
        };

        // Create new shard
        let mut shards = self.active_shards.write().await;
        shards.insert(
            new_shard_id,
            ShardInfo {
                shard_id: new_shard_id,
                shard_type: ShardType::Consensus,
                status: ShardStatus::Active,
                last_heartbeat: Instant::now(),
                performance_metrics: None,
                migration_state: None,
            },
        );

        tracing::info!(
            "Created new shard {} from split of shard {}",
            new_shard_id,
            source_shard
        );

        Ok(())
    }

    /// Execute shard merge operation
    async fn execute_merge_operation(
        &self,
        operation_id: Hash256,
        source_shards: Vec<u32>,
        target_shard: u32,
    ) -> Result<()> {
        tracing::info!(
            "Executing shard merge operation {:?} merging {:?} into {}",
            operation_id,
            source_shards,
            target_shard
        );

        let mut shards = self.active_shards.write().await;

        // Remove source shards
        for &source_shard in &source_shards {
            if source_shard != target_shard {
                shards.remove(&source_shard);
            }
        }

        Ok(())
    }

    /// Execute recovery operation for failed shard
    async fn execute_recovery_operation(
        &self,
        operation_id: Hash256,
        failed_shard: u32,
        _strategy: RecoveryStrategy,
    ) -> Result<()> {
        tracing::info!(
            "Executing recovery operation {:?} for failed shard {}",
            operation_id,
            failed_shard
        );

        let mut shards = self.active_shards.write().await;

        if let Some(shard_info) = shards.get_mut(&failed_shard) {
            // Mark shard as inactive and redistribute its load
            shard_info.status = ShardStatus::Inactive;
            tracing::warn!("Marked shard {} as inactive due to failure", failed_shard);
        }

        Ok(())
    }

    /// Trigger manual rebalancing
    pub async fn rebalance_shards(&mut self) -> Result<bool> {
        tracing::info!("Triggering manual shard rebalancing");

        // Check if already rebalancing
        {
            let rebalance_state = self.rebalancing_state.lock().await;
            if rebalance_state.is_rebalancing {
                tracing::info!("Rebalancing already in progress");
                return Ok(false);
            }
        }

        // Get all active shards
        let target_shards = {
            let shards = self.active_shards.read().await;
            shards.keys().cloned().collect::<Vec<_>>()
        };

        if !target_shards.is_empty() {
            self.queue_rebalance_operation(target_shards[0], RebalanceTrigger::Scheduled)
                .await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get coordinator metrics
    pub async fn get_coordinator_metrics(&self) -> CoordinatorMetrics {
        let metrics = self.metrics_collector.read().await;
        let mut coordinator_metrics = metrics.clone();

        // Update active migrations count
        let shards = self.active_shards.read().await;
        coordinator_metrics.active_migrations = shards
            .values()
            .filter(|shard| shard.status == ShardStatus::Migrating)
            .count();

        coordinator_metrics
    }

    /// Get current rebalancing status
    pub async fn get_rebalancing_status(&self) -> RebalancingStatus {
        let rebalance_state = self.rebalancing_state.lock().await;

        RebalancingStatus {
            is_active: rebalance_state.is_rebalancing,
            operation_id: rebalance_state.rebalance_id,
            progress: if rebalance_state.total_operations > 0 {
                rebalance_state.completed_operations as f64
                    / rebalance_state.total_operations as f64
            } else {
                0.0
            },
            started_at: rebalance_state.started_at,
            operations_completed: rebalance_state.completed_operations,
            operations_total: rebalance_state.total_operations,
        }
    }

    /// Get active shard information
    pub async fn get_shard_status(&self) -> HashMap<u32, ShardStatusInfo> {
        let shards = self.active_shards.read().await;

        shards
            .iter()
            .map(|(id, info)| {
                let status_info = ShardStatusInfo {
                    shard_id: *id,
                    shard_type: info.shard_type.clone(),
                    status: info.status.clone(),
                    last_heartbeat: info.last_heartbeat,
                    has_migration: info.migration_state.is_some(),
                    current_tps: info
                        .performance_metrics
                        .as_ref()
                        .map(|m| m.transactions_per_second)
                        .unwrap_or(0.0),
                };
                (*id, status_info)
            })
            .collect()
    }
}

/// Rebalancing status information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RebalancingStatus {
    pub is_active: bool,
    pub operation_id: Hash256,
    pub progress: f64,
    #[serde(skip)]
    #[serde(default)]
    pub started_at: Option<Instant>,
    pub operations_completed: usize,
    pub operations_total: usize,
}

impl Default for RebalancingStatus {
    fn default() -> Self {
        Self {
            is_active: false,
            operation_id: [0u8; 32],
            progress: 0.0,
            started_at: None,
            operations_completed: 0,
            operations_total: 0,
        }
    }
}

/// Shard status information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardStatusInfo {
    pub shard_id: u32,
    pub shard_type: ShardType,
    pub status: ShardStatus,
    #[serde(skip)]
    #[serde(default = "std::time::Instant::now")]
    pub last_heartbeat: Instant,
    pub has_migration: bool,
    pub current_tps: f64,
}

impl Default for ShardStatusInfo {
    fn default() -> Self {
        Self {
            shard_id: 0,
            shard_type: ShardType::Consensus,
            status: ShardStatus::Active,
            last_heartbeat: Instant::now(),
            has_migration: false,
            current_tps: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = ShardConfig::default();
        let coordinator = ShardCoordinator::new(config).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_operation_processing() {
        let config = ShardConfig::default();
        let coordinator = ShardCoordinator::new(config).await.unwrap();

        let processed = coordinator.process_operations().await;
        assert!(processed.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = ShardConfig::default();
        let coordinator = ShardCoordinator::new(config).await.unwrap();

        let metrics = coordinator.get_coordinator_metrics().await;
        assert_eq!(metrics.total_operations, 0);
    }
}
