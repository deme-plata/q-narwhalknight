// Q-NarwhalKnight State Sharding Implementation
// Partitioned state management for horizontal scaling

use crate::{ShardConfig, ShardId, ShardMetrics};
use anyhow::Result;
use q_types::{Hash256, StateKey, StateValue, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// State shard managing partition of global state
#[derive(Debug, Clone)]
pub struct StateShard {
    pub shard_id: u32,
    #[allow(dead_code)]
    config: ShardConfig,
    state_partition: Arc<RwLock<StatePartition>>,
    access_stats: Arc<RwLock<StateAccessStats>>,
    checkpoint_manager: CheckpointManager,
}

/// Partitioned state storage with versioning
#[derive(Debug)]
struct StatePartition {
    current_state: HashMap<StateKey, StateValue>,
    version_history: BTreeMap<u64, HashMap<StateKey, StateValue>>, // version -> state snapshot
    pending_updates: HashMap<StateKey, StateValue>,
    last_checkpoint: u64,
}

/// Access pattern statistics for optimization
#[derive(Debug, Clone)]
struct StateAccessStats {
    read_count: u64,
    write_count: u64,
    cache_hits: u64,
    cache_misses: u64,
    hot_keys: HashMap<StateKey, u32>, // key -> access frequency
    last_access_time: HashMap<StateKey, SystemTime>,
    total_operations: u64,
}

/// Checkpoint management for state recovery
#[derive(Debug, Clone)]
struct CheckpointManager {
    checkpoint_interval: u64,
    max_versions: usize,
    #[allow(dead_code)]
    compression_enabled: bool,
}

impl Default for StateAccessStats {
    fn default() -> Self {
        Self {
            read_count: 0,
            write_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            hot_keys: HashMap::new(),
            last_access_time: HashMap::new(),
            total_operations: 0,
        }
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self {
            checkpoint_interval: 1000, // Checkpoint every 1000 operations
            max_versions: 100,         // Keep 100 historical versions
            compression_enabled: true,
        }
    }
}

impl StateShard {
    /// Create new state shard for specific partition
    pub async fn new(shard_id: u32, config: ShardConfig) -> Result<Self> {
        tracing::info!("Creating state shard {} for partition management", shard_id);

        let state_partition = Arc::new(RwLock::new(StatePartition {
            current_state: HashMap::new(),
            version_history: BTreeMap::new(),
            pending_updates: HashMap::new(),
            last_checkpoint: 0,
        }));

        let access_stats = Arc::new(RwLock::new(StateAccessStats::default()));
        let checkpoint_manager = CheckpointManager::default();

        Ok(Self {
            shard_id,
            config,
            state_partition,
            access_stats,
            checkpoint_manager,
        })
    }

    /// Read state value for given key
    pub async fn read_state(&self, key: &StateKey) -> Result<Option<StateValue>> {
        let mut stats = self.access_stats.write().await;
        stats.read_count += 1;
        stats.total_operations += 1;

        // Update access frequency for hot key detection
        *stats.hot_keys.entry(key.clone()).or_insert(0) += 1;
        stats
            .last_access_time
            .insert(key.clone(), SystemTime::now());

        let partition = self.state_partition.read().await;

        // Check pending updates first
        if let Some(value) = partition.pending_updates.get(key) {
            stats.cache_hits += 1;
            return Ok(Some(value.clone()));
        }

        // Check current state
        if let Some(value) = partition.current_state.get(key) {
            stats.cache_hits += 1;
            Ok(Some(value.clone()))
        } else {
            stats.cache_misses += 1;
            Ok(None)
        }
    }

    /// Write state value for given key
    pub async fn write_state(&self, key: StateKey, value: StateValue) -> Result<()> {
        let mut stats = self.access_stats.write().await;
        stats.write_count += 1;
        stats.total_operations += 1;

        // Update access patterns
        *stats.hot_keys.entry(key.clone()).or_insert(0) += 1;
        stats
            .last_access_time
            .insert(key.clone(), SystemTime::now());
        drop(stats);

        // Add to pending updates (will be committed in batch)
        let mut partition = self.state_partition.write().await;
        partition.pending_updates.insert(key, value);

        // Check if we need to checkpoint
        if partition.pending_updates.len() >= self.checkpoint_manager.checkpoint_interval as usize {
            self.create_checkpoint_internal(&mut partition).await?;
        }

        Ok(())
    }

    /// Apply a batch of state updates atomically
    pub async fn apply_state_batch(&self, updates: HashMap<StateKey, StateValue>) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "Applying state batch with {} updates to shard {}",
            updates.len(),
            self.shard_id
        );

        let mut partition = self.state_partition.write().await;
        let mut stats = self.access_stats.write().await;

        // Apply all updates to pending state
        for (key, value) in updates {
            partition.pending_updates.insert(key.clone(), value);
            *stats.hot_keys.entry(key.clone()).or_insert(0) += 1;
            stats.last_access_time.insert(key, SystemTime::now());
        }

        stats.write_count += partition.pending_updates.len() as u64;
        stats.total_operations += partition.pending_updates.len() as u64;
        drop(stats);

        // Commit pending updates if batch is large enough
        if partition.pending_updates.len() >= self.checkpoint_manager.checkpoint_interval as usize {
            self.create_checkpoint_internal(&mut partition).await?;
        }

        Ok(())
    }

    /// Commit all pending updates to current state
    pub async fn commit_pending_updates(&self) -> Result<u64> {
        let mut partition = self.state_partition.write().await;

        if partition.pending_updates.is_empty() {
            return Ok(partition.last_checkpoint);
        }

        let version = self.create_checkpoint_internal(&mut partition).await?;
        Ok(version)
    }

    /// Create checkpoint and commit pending updates
    async fn create_checkpoint_internal(&self, partition: &mut StatePartition) -> Result<u64> {
        if partition.pending_updates.is_empty() {
            return Ok(partition.last_checkpoint);
        }

        // Generate new version number
        let new_version = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Apply pending updates to current state
        for (key, value) in partition.pending_updates.drain() {
            partition.current_state.insert(key, value);
        }

        // Create historical snapshot if versioning enabled
        if self.checkpoint_manager.max_versions > 0 {
            partition
                .version_history
                .insert(new_version, partition.current_state.clone());

            // Cleanup old versions if we exceed max_versions
            while partition.version_history.len() > self.checkpoint_manager.max_versions {
                if let Some((oldest_version, _)) = partition.version_history.iter().next() {
                    let oldest_version = *oldest_version;
                    partition.version_history.remove(&oldest_version);
                } else {
                    break;
                }
            }
        }

        partition.last_checkpoint = new_version;

        tracing::debug!(
            "Created state checkpoint {} for shard {} with {} keys",
            new_version,
            self.shard_id,
            partition.current_state.len()
        );

        Ok(new_version)
    }

    /// Get historical state at specific version
    pub async fn get_state_at_version(
        &self,
        version: u64,
    ) -> Result<Option<HashMap<StateKey, StateValue>>> {
        let partition = self.state_partition.read().await;
        Ok(partition.version_history.get(&version).cloned())
    }

    /// Get hot keys (most frequently accessed)
    pub async fn get_hot_keys(&self, limit: usize) -> Result<Vec<(StateKey, u32)>> {
        let stats = self.access_stats.read().await;

        let mut hot_keys: Vec<_> = stats
            .hot_keys
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        hot_keys.sort_by(|a, b| b.1.cmp(&a.1));
        hot_keys.truncate(limit);

        Ok(hot_keys)
    }

    /// Get current shard performance metrics
    pub async fn get_metrics(&self) -> Result<ShardMetrics> {
        let stats = self.access_stats.read().await;
        let partition = self.state_partition.read().await;

        // Calculate cache hit ratio
        let total_reads = stats.cache_hits + stats.cache_misses;
        let cache_hit_ratio = if total_reads > 0 {
            stats.cache_hits as f64 / total_reads as f64
        } else {
            1.0
        };

        // Estimate TPS based on recent operations
        let operations_per_second = if stats.total_operations > 0 {
            // Simplified calculation - in real implementation would track time windows
            stats.total_operations as f64 / 60.0 // Assume 60 second window
        } else {
            0.0
        };

        // Estimate resource usage
        let state_size = partition.current_state.len() + partition.pending_updates.len();
        let memory_usage_mb = (state_size * 1024) as f64 / (1024.0 * 1024.0); // Rough estimate
        let cpu_utilization = (operations_per_second / 10000.0).min(1.0); // Rough estimate

        Ok(ShardMetrics {
            shard_id: ShardId::State(self.shard_id),
            transactions_per_second: operations_per_second,
            average_latency_ms: 1.0, // State operations are typically very fast
            cpu_utilization,
            memory_usage_mb,
            active_connections: 1, // Simplified
            queue_depth: partition.pending_updates.len(),
        })
    }

    /// Get state shard size (number of keys)
    pub async fn get_state_size(&self) -> usize {
        let partition = self.state_partition.read().await;
        partition.current_state.len()
    }

    /// Get pending updates count
    pub async fn get_pending_count(&self) -> usize {
        let partition = self.state_partition.read().await;
        partition.pending_updates.len()
    }

    /// Check if shard needs rebalancing based on size or access patterns
    pub async fn needs_rebalancing(&self) -> bool {
        let state_size = self.get_state_size().await;
        let stats = self.access_stats.read().await;

        // Rebalance if state is too large or access patterns are skewed
        let size_threshold = 1000000; // 1M keys per shard
        let hot_key_threshold = stats.total_operations / 100; // Top 1% of accesses

        if state_size > size_threshold {
            return true;
        }

        // Check for hot key concentration
        let hot_key_count = stats
            .hot_keys
            .values()
            .filter(|&&count| count as u64 > hot_key_threshold)
            .count();
        hot_key_count > 10 // If more than 10 extremely hot keys
    }

    /// Export state for migration/rebalancing
    pub async fn export_state(&self) -> Result<HashMap<StateKey, StateValue>> {
        let partition = self.state_partition.read().await;
        let mut exported_state = partition.current_state.clone();

        // Include pending updates
        for (key, value) in &partition.pending_updates {
            exported_state.insert(key.clone(), value.clone());
        }

        Ok(exported_state)
    }

    /// Import state during migration/rebalancing
    pub async fn import_state(&self, state: HashMap<StateKey, StateValue>) -> Result<()> {
        let mut partition = self.state_partition.write().await;

        // Replace current state with imported state
        partition.current_state = state;
        partition.pending_updates.clear();

        // Create checkpoint for the imported state
        let version = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        partition.last_checkpoint = version;

        if self.checkpoint_manager.max_versions > 0 {
            let current_state_clone = partition.current_state.clone();
            partition
                .version_history
                .insert(version, current_state_clone);
        }

        tracing::info!(
            "Imported {} state keys into shard {}",
            partition.current_state.len(),
            self.shard_id
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::{create_test_state_key, create_test_state_value};

    #[tokio::test]
    async fn test_state_shard_creation() {
        let config = ShardConfig::default();
        let shard = StateShard::new(0, config).await;
        assert!(shard.is_ok());
    }

    #[tokio::test]
    async fn test_state_read_write() {
        let config = ShardConfig::default();
        let shard = StateShard::new(0, config).await.unwrap();

        let key = create_test_state_key();
        let value = create_test_state_value();

        // Write state
        let result = shard.write_state(key.clone(), value.clone()).await;
        assert!(result.is_ok());

        // Read state
        let read_result = shard.read_state(&key).await;
        assert!(read_result.is_ok());
        assert_eq!(read_result.unwrap(), Some(value));
    }

    #[tokio::test]
    async fn test_batch_updates() {
        let config = ShardConfig::default();
        let shard = StateShard::new(0, config).await.unwrap();

        let mut updates = HashMap::new();
        for i in 0..10 {
            updates.insert(create_test_state_key(), create_test_state_value());
        }

        let result = shard.apply_state_batch(updates).await;
        assert!(result.is_ok());

        let state_size = shard.get_state_size().await;
        assert!(state_size > 0);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = ShardConfig::default();
        let shard = StateShard::new(0, config).await.unwrap();

        let metrics = shard.get_metrics().await;
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.shard_id, ShardId::State(0));
    }
}
