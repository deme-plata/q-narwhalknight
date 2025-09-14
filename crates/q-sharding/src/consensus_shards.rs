// Q-NarwhalKnight Consensus Sharding Implementation
// Individual consensus shard management for horizontal scaling

use crate::{ShardConfig, ShardId, ShardMetrics};
use anyhow::Result;
use q_dag_knight::DAGKnightConsensus;
use q_types::{Hash256, Transaction, Vertex, VertexId};
use sha3;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing;

/// Simplified mempool for sharding
struct ShardMempool {
    transactions: Vec<Transaction>,
    max_size: usize,
}

impl ShardMempool {
    fn new() -> Self {
        Self {
            transactions: Vec::new(),
            max_size: 10000,
        }
    }

    fn add_transaction(&mut self, tx: Transaction) -> Result<()> {
        if self.transactions.len() < self.max_size {
            self.transactions.push(tx);
        }
        Ok(())
    }

    fn get_pending_transactions(&mut self, count: usize) -> Result<Vec<Transaction>> {
        let actual_count = count.min(self.transactions.len());
        let txs: Vec<_> = self.transactions.drain(0..actual_count).collect();
        Ok(txs)
    }
}

/// Individual consensus shard handling subset of transactions
#[derive(Clone)]
pub struct ConsensusShard {
    pub shard_id: u32,
    config: ShardConfig,
    consensus: Arc<RwLock<DAGKnightConsensus>>,
    mempool: Arc<RwLock<ShardMempool>>,
    transaction_queue: Arc<RwLock<VecDeque<Transaction>>>,
    processing_stats: Arc<RwLock<ShardProcessingStats>>,
}

#[derive(Debug, Clone)]
struct ShardProcessingStats {
    transactions_processed: u64,
    total_processing_time_ms: u64,
    vertices_created: u64,
    last_batch_time: Option<Instant>,
    average_latency_ms: f64,
}

impl Default for ShardProcessingStats {
    fn default() -> Self {
        Self {
            transactions_processed: 0,
            total_processing_time_ms: 0,
            vertices_created: 0,
            last_batch_time: None,
            average_latency_ms: 0.0,
        }
    }
}

impl ConsensusShard {
    /// Create new consensus shard
    pub async fn new(shard_id: u32, config: ShardConfig) -> Result<Self> {
        tracing::info!("Creating consensus shard {}", shard_id);

        // Create node ID as a hash of the shard identifier
        let node_id = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(format!("shard_{}", shard_id).as_bytes());
            let hash = hasher.finalize();
            let mut array = [0u8; 32];
            array.copy_from_slice(&hash[..32]);
            array
        };

        let consensus = Arc::new(RwLock::new(DAGKnightConsensus::new(node_id, 1).await?));

        // For sharding, we'll use a simplified mempool wrapper
        // TODO: Integrate with full ProductionMempool in Phase 2
        let mempool = Arc::new(RwLock::new(ShardMempool::new()));

        let transaction_queue = Arc::new(RwLock::new(VecDeque::new()));
        let processing_stats = Arc::new(RwLock::new(ShardProcessingStats::default()));

        Ok(Self {
            shard_id,
            config,
            consensus,
            mempool,
            transaction_queue,
            processing_stats,
        })
    }

    /// Process a batch of transactions assigned to this shard
    pub async fn process_transactions(
        &self,
        transactions: Vec<Transaction>,
    ) -> Result<Vec<VertexId>> {
        let start_time = Instant::now();
        let tx_count = transactions.len();

        tracing::debug!(
            "Shard {} processing {} transactions",
            self.shard_id,
            tx_count
        );

        // Add transactions to queue
        {
            let mut queue = self.transaction_queue.write().await;
            for tx in transactions {
                queue.push_back(tx);
            }
        }

        // Process transactions in batches
        let mut vertex_ids = Vec::new();
        let batch_size = self.config.max_tx_per_shard_batch.min(tx_count);

        while vertex_ids.len() < tx_count {
            let batch = self.get_next_batch(batch_size).await?;
            if batch.is_empty() {
                break;
            }

            let batch_vertex_ids = self.process_transaction_batch(batch).await?;
            vertex_ids.extend(batch_vertex_ids);
        }

        // Update processing statistics
        let processing_time = start_time.elapsed();
        self.update_stats(tx_count, processing_time).await;

        tracing::debug!(
            "Shard {} completed processing {} transactions in {}ms",
            self.shard_id,
            tx_count,
            processing_time.as_millis()
        );

        Ok(vertex_ids)
    }

    /// Get next batch of transactions from queue
    async fn get_next_batch(&self, max_size: usize) -> Result<Vec<Transaction>> {
        let mut queue = self.transaction_queue.write().await;
        let mut batch = Vec::new();

        for _ in 0..max_size {
            if let Some(tx) = queue.pop_front() {
                batch.push(tx);
            } else {
                break;
            }
        }

        Ok(batch)
    }

    /// Process a single batch of transactions through DAG-Knight consensus
    async fn process_transaction_batch(&self, batch: Vec<Transaction>) -> Result<Vec<VertexId>> {
        let mut vertex_ids = Vec::new();

        // Add transactions to mempool first
        {
            let mut mempool = self.mempool.write().await;
            for tx in &batch {
                mempool.add_transaction(tx.clone())?;
            }
        }

        // Create vertex with this batch
        let mut consensus = self.consensus.write().await;

        // Get transactions from mempool and create vertex
        let mempool_txs = {
            let mut mempool = self.mempool.write().await;
            mempool.get_pending_transactions(batch.len())?
        };

        if !mempool_txs.is_empty() {
            // For sharding, create a simplified vertex directly
            // TODO: Integrate with full DAG-Knight vertex creation in Phase 2
            let vertex_id = {
                use sha3::{Digest, Sha3_256};
                let mut hasher = Sha3_256::new();
                hasher.update(
                    format!("shard_{}_batch_{}", self.shard_id, mempool_txs.len()).as_bytes(),
                );
                let hash = hasher.finalize();
                let mut array = [0u8; 32];
                array.copy_from_slice(&hash[..32]);
                array
            };

            vertex_ids.push(vertex_id);

            // Update stats
            let mut stats = self.processing_stats.write().await;
            stats.vertices_created += 1;

            tracing::debug!(
                "Shard {} created vertex {:?} with {} transactions",
                self.shard_id,
                vertex_id,
                mempool_txs.len()
            );
        }

        Ok(vertex_ids)
    }

    /// Update shard processing statistics
    async fn update_stats(&self, tx_count: usize, processing_time: std::time::Duration) {
        let mut stats = self.processing_stats.write().await;

        stats.transactions_processed += tx_count as u64;
        stats.total_processing_time_ms += processing_time.as_millis() as u64;
        stats.last_batch_time = Some(Instant::now());

        // Calculate rolling average latency
        if stats.transactions_processed > 0 {
            stats.average_latency_ms =
                stats.total_processing_time_ms as f64 / stats.transactions_processed as f64;
        }
    }

    /// Get current shard performance metrics
    pub async fn get_metrics(&self) -> Result<ShardMetrics> {
        let stats = self.processing_stats.read().await;

        // Calculate current TPS based on recent activity
        let current_tps = if let Some(last_batch) = stats.last_batch_time {
            let seconds_since = last_batch.elapsed().as_secs_f64();
            if seconds_since > 0.0 && seconds_since < 60.0 {
                stats.transactions_processed as f64 / seconds_since
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Estimate resource usage (simplified for now)
        let queue_len = self.transaction_queue.read().await.len();
        let cpu_utilization = (current_tps / 1000.0).min(1.0); // Rough estimate
        let memory_usage_mb =
            (queue_len * 1024 + stats.vertices_created as usize * 2048) as f64 / (1024.0 * 1024.0);

        Ok(ShardMetrics {
            shard_id: ShardId::Consensus(self.shard_id),
            transactions_per_second: current_tps,
            average_latency_ms: stats.average_latency_ms,
            cpu_utilization,
            memory_usage_mb,
            active_connections: 1, // Simplified
            queue_depth: queue_len,
        })
    }

    /// Get queue depth for load balancing
    pub async fn get_queue_depth(&self) -> usize {
        self.transaction_queue.read().await.len()
    }

    /// Check if shard is overloaded
    pub async fn is_overloaded(&self) -> bool {
        let queue_depth = self.get_queue_depth().await;
        queue_depth > self.config.max_tx_per_shard_batch * 10
    }

    /// Get total transactions processed by this shard
    pub async fn get_total_transactions(&self) -> u64 {
        self.processing_stats.read().await.transactions_processed
    }

    /// Get shard processing efficiency
    pub async fn get_efficiency_score(&self) -> f64 {
        let stats = self.processing_stats.read().await;

        if stats.transactions_processed == 0 {
            return 1.0;
        }

        // Efficiency = processed / (processed + queue_depth)
        let queue_depth = self.transaction_queue.read().await.len() as u64;
        stats.transactions_processed as f64 / (stats.transactions_processed + queue_depth) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::create_test_transaction;

    #[tokio::test]
    async fn test_consensus_shard_creation() {
        let config = ShardConfig::default();
        let shard = ConsensusShard::new(0, config).await;
        assert!(shard.is_ok());
    }

    #[tokio::test]
    async fn test_transaction_processing() {
        let config = ShardConfig::default();
        let shard = ConsensusShard::new(0, config).await.unwrap();

        let transactions = vec![create_test_transaction(), create_test_transaction()];

        let result = shard.process_transactions(transactions).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = ShardConfig::default();
        let shard = ConsensusShard::new(0, config).await.unwrap();

        let metrics = shard.get_metrics().await;
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.shard_id, ShardId::Consensus(0));
        assert!(metrics.transactions_per_second >= 0.0);
    }
}
