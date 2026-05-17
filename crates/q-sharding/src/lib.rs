// Q-NarwhalKnight Sharding System
// High-performance horizontal scaling for DAG-based consensus

pub mod consensus_shards;
pub mod cross_shard_bridge;
pub mod load_balancer;
pub mod metrics;
pub mod shard_coordinator;
pub mod state_shards;

use anyhow::Result;
use q_types::{Hash256, NodeId, Transaction, Vertex, VertexId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main sharding configuration
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of consensus shards
    pub consensus_shards: u32,
    /// Number of state shards
    pub state_shards: u32,
    /// Shard rebalancing threshold (0.0-1.0)
    pub rebalance_threshold: f64,
    /// Maximum transactions per shard per batch
    pub max_tx_per_shard_batch: usize,
    /// Cross-shard communication timeout
    pub cross_shard_timeout_ms: u64,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            consensus_shards: 4,      // Start with 4 consensus shards
            state_shards: 8,          // 8 state shards for better distribution
            rebalance_threshold: 0.8, // Rebalance when 80% capacity reached
            max_tx_per_shard_batch: 1000,
            cross_shard_timeout_ms: 100, // 100ms cross-shard timeout
        }
    }
}

/// Shard identifier and type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShardId {
    Consensus(u32),
    State(u32),
}

impl ShardId {
    /// Get the numeric ID of the shard
    pub fn id(&self) -> u32 {
        match self {
            ShardId::Consensus(id) => *id,
            ShardId::State(id) => *id,
        }
    }

    /// Convert to u32 for cross-shard result compatibility
    pub fn to_u32(&self) -> u32 {
        match self {
            ShardId::Consensus(id) => *id,
            ShardId::State(id) => *id + 1000, // Offset state shards to avoid collisions
        }
    }
}

/// Transaction distribution strategy
#[derive(Debug, Clone)]
pub enum ShardingStrategy {
    /// Hash-based partitioning (most common)
    HashBased,
    /// Load-based dynamic assignment
    LoadBased,
    /// Geographic/network proximity based
    ProximityBased,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Shard performance metrics
#[derive(Debug, Clone)]
pub struct ShardMetrics {
    pub shard_id: ShardId,
    pub transactions_per_second: f64,
    pub average_latency_ms: f64,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub active_connections: u32,
    pub queue_depth: usize,
}

/// Main sharding engine
pub struct ShardingEngine {
    config: ShardConfig,
    strategy: ShardingStrategy,
    consensus_shards: Arc<RwLock<HashMap<u32, consensus_shards::ConsensusShard>>>,
    state_shards: Arc<RwLock<HashMap<u32, state_shards::StateShard>>>,
    load_balancer: load_balancer::LoadBalancer,
    coordinator: shard_coordinator::ShardCoordinator,
    metrics: metrics::ShardMetricsCollector,
}

impl ShardingEngine {
    /// Create new sharding engine
    pub async fn new(config: ShardConfig) -> Result<Self> {
        let consensus_shards = Arc::new(RwLock::new(HashMap::new()));
        let state_shards = Arc::new(RwLock::new(HashMap::new()));

        // Initialize consensus shards
        {
            let mut shards = consensus_shards.write().await;
            for shard_id in 0..config.consensus_shards {
                let shard = consensus_shards::ConsensusShard::new(shard_id, config.clone()).await?;
                shards.insert(shard_id, shard);
            }
        }

        // Initialize state shards
        {
            let mut shards = state_shards.write().await;
            for shard_id in 0..config.state_shards {
                let shard = state_shards::StateShard::new(shard_id, config.clone()).await?;
                shards.insert(shard_id, shard);
            }
        }

        let load_balancer = load_balancer::LoadBalancer::new(config.clone()).await?;
        let coordinator = shard_coordinator::ShardCoordinator::new(config.clone()).await?;
        let metrics = metrics::ShardMetricsCollector::new();

        Ok(Self {
            config,
            strategy: ShardingStrategy::HashBased,
            consensus_shards,
            state_shards,
            load_balancer,
            coordinator,
            metrics,
        })
    }

    /// Route transaction to appropriate consensus shard
    pub async fn route_transaction(&self, tx: &Transaction) -> Result<u32> {
        match self.strategy {
            ShardingStrategy::HashBased => Ok(self.hash_based_routing(&tx.hash())),
            ShardingStrategy::LoadBased => self.load_balancer.select_least_loaded_shard().await,
            _ => {
                // Fallback to hash-based
                Ok(self.hash_based_routing(&tx.hash()))
            }
        }
    }

    /// Process transaction batch across shards
    pub async fn process_transaction_batch(&self, transactions: Vec<Transaction>) -> Result<()> {
        // Group transactions by shard
        let mut shard_batches: HashMap<u32, Vec<Transaction>> = HashMap::new();

        for tx in transactions {
            let shard_id = self.route_transaction(&tx).await?;
            shard_batches
                .entry(shard_id)
                .or_insert_with(Vec::new)
                .push(tx);
        }

        // Process each shard batch in parallel
        let mut handles = Vec::new();
        let shards = self.consensus_shards.read().await;

        for (shard_id, batch) in shard_batches {
            if let Some(shard) = shards.get(&shard_id) {
                let shard_clone = shard.clone();
                let handle =
                    tokio::spawn(async move { shard_clone.process_transactions(batch).await });
                handles.push(handle);
            }
        }

        // Wait for all shards to complete
        for handle in handles {
            handle.await??;
        }

        Ok(())
    }

    /// Get shard for state key
    pub fn get_state_shard(&self, key: &Hash256) -> u32 {
        (hash_to_u64(key) % self.config.state_shards as u64) as u32
    }

    /// Hash-based transaction routing
    fn hash_based_routing(&self, tx_hash: &Hash256) -> u32 {
        (hash_to_u64(tx_hash) % self.config.consensus_shards as u64) as u32
    }

    /// Get current shard metrics
    pub async fn get_shard_metrics(&self) -> Result<Vec<ShardMetrics>> {
        self.metrics.collect_all_shard_metrics().await
    }

    /// Trigger shard rebalancing if needed
    pub async fn rebalance_if_needed(&mut self) -> Result<bool> {
        let metrics = self.get_shard_metrics().await?;

        // Check if any shard is over threshold
        for metric in &metrics {
            if metric.cpu_utilization > self.config.rebalance_threshold {
                tracing::info!(
                    "Triggering shard rebalancing due to high load on {:?}",
                    metric.shard_id
                );
                return self.coordinator.rebalance_shards().await;
            }
        }

        Ok(false)
    }

    /// Get total system TPS across all shards
    pub async fn get_total_tps(&self) -> Result<f64> {
        let metrics = self.get_shard_metrics().await?;
        Ok(metrics.iter().map(|m| m.transactions_per_second).sum())
    }
}

/// Shard assignment for a specific transaction
#[derive(Debug, Clone)]
pub struct ShardAssignment {
    pub transaction_hash: Hash256,
    pub consensus_shard: u32,
    pub state_shard: u32,
    pub assigned_at: std::time::Instant,
}

/// Cross-shard operation result
#[derive(Debug, Clone)]
pub struct CrossShardResult {
    pub operation_id: Hash256,
    pub involved_shards: Vec<u32>,
    pub success: bool,
    pub execution_time_ms: u64,
    pub data_transferred_bytes: u64,
}

/// Utility function to convert Hash256 to u64 for shard routing
fn hash_to_u64(hash: &Hash256) -> u64 {
    // Take first 8 bytes of the hash and convert to u64
    u64::from_be_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ])
}

/// Generate a random Hash256 for operations that need unique IDs
fn generate_random_hash() -> Hash256 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 32];
    rng.fill(&mut bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sharding_engine_creation() {
        let config = ShardConfig::default();
        let engine = ShardingEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_hash_based_routing() {
        let config = ShardConfig::default();
        let engine = ShardingEngine::new(config).await.unwrap();

        // Test consistent routing
        let hash1 = [1u8; 32];
        let hash2 = [1u8; 32];

        let shard1 = engine.hash_based_routing(&hash1);
        let shard2 = engine.hash_based_routing(&hash2);

        assert_eq!(shard1, shard2);
    }

    #[tokio::test]
    async fn test_shard_distribution() {
        let config = ShardConfig::default();
        let engine = ShardingEngine::new(config).await.unwrap();

        // Test that different hashes go to different shards
        let mut shard_counts = HashMap::new();

        for i in 0..1000 {
            let mut bytes = [0u8; 32];
            bytes[0..8].copy_from_slice(&i.to_be_bytes());
            let hash = bytes;
            let shard = engine.hash_based_routing(&hash);
            *shard_counts.entry(shard).or_insert(0) += 1;
        }

        // Should distribute across multiple shards
        assert!(shard_counts.len() > 1);
    }
}
