// Q-NarwhalKnight Cross-Shard Communication Bridge
// Inter-shard messaging and coordination protocols

use crate::{CrossShardResult, ShardId};
use anyhow::Result;
use futures::future::join_all;
use q_types::{Hash256, NodeId, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock};

/// Cross-shard communication bridge for coordinating operations across shards
#[derive(Debug)]
pub struct CrossShardBridge {
    bridge_id: String,
    message_channels: Arc<RwLock<HashMap<ShardId, mpsc::UnboundedSender<CrossShardMessage>>>>,
    pending_operations: Arc<RwLock<HashMap<Hash256, PendingCrossShardOperation>>>,
    operation_timeout: Duration,
    metrics: Arc<RwLock<CrossShardMetrics>>,
}

/// Messages exchanged between shards
#[derive(Debug, Clone)]
pub enum CrossShardMessage {
    /// Request to validate transaction dependencies across shards
    ValidateTransactionDeps {
        operation_id: Hash256,
        transaction: Transaction,
        dependent_shards: Vec<ShardId>,
        // Note: response_channel is handled separately via operation tracking
    },

    /// Response to validation request
    ValidationResponse {
        operation_id: Hash256,
        shard_id: ShardId,
        is_valid: bool,
        validation_data: Vec<u8>,
    },

    /// Request state data from another shard
    StateRequest {
        operation_id: Hash256,
        state_keys: Vec<String>,
        requesting_shard: ShardId,
    },

    /// Response with requested state data
    StateResponse {
        operation_id: Hash256,
        state_data: HashMap<String, Vec<u8>>,
        responding_shard: ShardId,
    },

    /// Coordinate atomic commit across multiple shards
    AtomicCommitPrepare {
        operation_id: Hash256,
        coordinator_shard: ShardId,
        participant_shards: Vec<ShardId>,
        transaction_batch: Vec<Transaction>,
    },

    /// Response to atomic commit preparation
    AtomicCommitResponse {
        operation_id: Hash256,
        shard_id: ShardId,
        vote: CommitVote,
    },

    /// Final commit decision
    AtomicCommitDecision {
        operation_id: Hash256,
        decision: CommitDecision,
    },

    /// Shard rebalancing coordination
    RebalanceRequest {
        operation_id: Hash256,
        source_shard: ShardId,
        target_shard: ShardId,
        data_to_migrate: Vec<u8>,
    },

    /// Heartbeat for shard liveness detection
    Heartbeat {
        shard_id: ShardId,
        timestamp: u64,
        load_info: ShardLoadInfo,
    },
}

/// Atomic commit voting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommitVote {
    Prepare, // Ready to commit
    Abort,   // Cannot commit
}

/// Atomic commit decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommitDecision {
    Commit, // All shards voted prepare
    Abort,  // At least one shard voted abort
}

/// Load information for shard balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardLoadInfo {
    pub transactions_per_second: f64,
    pub queue_depth: usize,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
}

/// Pending cross-shard operation tracking
#[derive(Debug)]
struct PendingCrossShardOperation {
    operation_id: Hash256,
    operation_type: CrossShardOperationType,
    involved_shards: Vec<ShardId>,
    responses_received: HashMap<ShardId, CrossShardMessage>,
    created_at: Instant,
    timeout_at: Instant,
    completion_callback: Option<oneshot::Sender<CrossShardResult>>,
}

/// Types of cross-shard operations
#[derive(Debug, Clone)]
pub enum CrossShardOperationType {
    TransactionValidation,
    StateQuery,
    AtomicCommit,
    Rebalancing,
}

/// Cross-shard bridge performance metrics
#[derive(Debug, Clone, Default)]
struct CrossShardMetrics {
    messages_sent: u64,
    messages_received: u64,
    operations_completed: u64,
    operations_timed_out: u64,
    average_operation_time_ms: f64,
    active_operations: usize,
}

impl CrossShardBridge {
    /// Create new cross-shard communication bridge
    pub fn new(bridge_id: String, operation_timeout: Duration) -> Self {
        tracing::info!("Creating cross-shard bridge: {}", bridge_id);

        Self {
            bridge_id,
            message_channels: Arc::new(RwLock::new(HashMap::new())),
            pending_operations: Arc::new(RwLock::new(HashMap::new())),
            operation_timeout,
            metrics: Arc::new(RwLock::new(CrossShardMetrics::default())),
        }
    }

    /// Register a shard with the bridge
    pub async fn register_shard(
        &self,
        shard_id: ShardId,
        sender: mpsc::UnboundedSender<CrossShardMessage>,
    ) {
        let mut channels = self.message_channels.write().await;
        channels.insert(shard_id, sender);

        tracing::info!("Registered shard {:?} with cross-shard bridge", shard_id);
    }

    /// Unregister a shard from the bridge
    pub async fn unregister_shard(&self, shard_id: &ShardId) {
        let mut channels = self.message_channels.write().await;
        channels.remove(shard_id);

        tracing::info!("Unregistered shard {:?} from cross-shard bridge", shard_id);
    }

    /// Send message to specific shard
    pub async fn send_message(
        &self,
        target_shard: ShardId,
        message: CrossShardMessage,
    ) -> Result<()> {
        let channels = self.message_channels.read().await;

        if let Some(sender) = channels.get(&target_shard) {
            sender.send(message)?;

            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.messages_sent += 1;

            Ok(())
        } else {
            anyhow::bail!("Shard {:?} not registered with bridge", target_shard);
        }
    }

    /// Broadcast message to all registered shards
    pub async fn broadcast_message(&self, message: CrossShardMessage) -> Result<()> {
        let channels = self.message_channels.read().await;

        for sender in channels.values() {
            sender.send(message.clone())?;
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.messages_sent += channels.len() as u64;

        Ok(())
    }

    /// Coordinate atomic commit across multiple shards
    pub async fn coordinate_atomic_commit(
        &self,
        transaction_batch: Vec<Transaction>,
        participant_shards: Vec<ShardId>,
    ) -> Result<CrossShardResult> {
        let operation_id = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(&uuid::Uuid::new_v4().to_string().as_bytes());
            let hash = hasher.finalize();
            let mut array = [0u8; 32];
            array.copy_from_slice(&hash[..32]);
            array
        };
        let coordinator_shard = ShardId::Consensus(0); // Use shard 0 as coordinator
        let start_time = Instant::now();

        tracing::debug!(
            "Starting atomic commit operation {:?} across {} shards",
            operation_id,
            participant_shards.len()
        );

        // Create completion channel
        let (completion_tx, completion_rx) = oneshot::channel();

        // Register pending operation
        {
            let mut pending = self.pending_operations.write().await;
            pending.insert(
                operation_id,
                PendingCrossShardOperation {
                    operation_id,
                    operation_type: CrossShardOperationType::AtomicCommit,
                    involved_shards: participant_shards.clone(),
                    responses_received: HashMap::new(),
                    created_at: start_time,
                    timeout_at: start_time + self.operation_timeout,
                    completion_callback: Some(completion_tx),
                },
            );
        }

        // Send prepare message to all participants
        let prepare_message = CrossShardMessage::AtomicCommitPrepare {
            operation_id,
            coordinator_shard,
            participant_shards: participant_shards.clone(),
            transaction_batch: transaction_batch.clone(),
        };

        for shard_id in &participant_shards {
            self.send_message(*shard_id, prepare_message.clone())
                .await?;
        }

        // Wait for completion or timeout
        match tokio::time::timeout(self.operation_timeout, completion_rx).await {
            Ok(Ok(result)) => {
                tracing::debug!(
                    "Atomic commit operation {:?} completed successfully",
                    operation_id
                );
                Ok(result)
            }
            Ok(Err(_)) => {
                tracing::error!(
                    "Atomic commit operation {:?} completion channel closed",
                    operation_id
                );
                anyhow::bail!("Operation completion channel closed")
            }
            Err(_) => {
                tracing::error!("Atomic commit operation {:?} timed out", operation_id);
                self.cleanup_timed_out_operation(operation_id).await;
                anyhow::bail!("Atomic commit operation timed out")
            }
        }
    }

    /// Handle incoming cross-shard message
    pub async fn handle_message(&self, message: CrossShardMessage) -> Result<()> {
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.messages_received += 1;
        }

        match message {
            CrossShardMessage::ValidationResponse {
                operation_id,
                shard_id,
                is_valid,
                ..
            } => {
                self.handle_validation_response(operation_id, shard_id, is_valid)
                    .await?;
            }

            CrossShardMessage::StateResponse {
                operation_id,
                state_data,
                responding_shard,
            } => {
                self.handle_state_response(operation_id, responding_shard, state_data)
                    .await?;
            }

            CrossShardMessage::AtomicCommitResponse {
                operation_id,
                shard_id,
                vote,
            } => {
                self.handle_atomic_commit_response(operation_id, shard_id, vote)
                    .await?;
            }

            CrossShardMessage::Heartbeat {
                shard_id,
                load_info,
                ..
            } => {
                self.handle_heartbeat(shard_id, load_info).await?;
            }

            _ => {
                tracing::debug!("Received cross-shard message: {:?}", message);
            }
        }

        Ok(())
    }

    /// Handle atomic commit response from participant shard
    async fn handle_atomic_commit_response(
        &self,
        operation_id: Hash256,
        shard_id: ShardId,
        vote: CommitVote,
    ) -> Result<()> {
        let mut pending = self.pending_operations.write().await;

        if let Some(operation) = pending.get_mut(&operation_id) {
            // Record the response
            let response_message = CrossShardMessage::AtomicCommitResponse {
                operation_id,
                shard_id,
                vote: vote.clone(),
            };
            operation
                .responses_received
                .insert(shard_id, response_message);

            // Check if all participants have responded
            if operation.responses_received.len() == operation.involved_shards.len() {
                // Determine commit decision
                let mut decision = CommitDecision::Commit;
                for (_, response) in &operation.responses_received {
                    if let CrossShardMessage::AtomicCommitResponse {
                        vote: CommitVote::Abort,
                        ..
                    } = response
                    {
                        decision = CommitDecision::Abort;
                        break;
                    }
                }

                // Send decision to all participants
                let decision_message = CrossShardMessage::AtomicCommitDecision {
                    operation_id,
                    decision: decision.clone(),
                };

                // Extract needed data before dropping the lock
                let involved_shards = operation.involved_shards.clone();
                let created_at = operation.created_at;
                let completion_tx = operation.completion_callback.take();

                drop(pending); // Release lock before sending messages

                for participant in &involved_shards {
                    self.send_message(*participant, decision_message.clone())
                        .await?;
                }

                // Complete the operation
                let execution_time = created_at.elapsed().as_millis() as u64;
                let data_transferred = 1024; // Simplified

                let result = CrossShardResult {
                    operation_id,
                    involved_shards: involved_shards.iter().map(|s| s.to_u32()).collect(),
                    success: matches!(decision, CommitDecision::Commit),
                    execution_time_ms: execution_time,
                    data_transferred_bytes: data_transferred,
                };

                // Send completion signal
                if let Some(completion_tx) = completion_tx {
                    let _ = completion_tx.send(result);
                }

                // Update metrics
                let mut metrics = self.metrics.write().await;
                metrics.operations_completed += 1;
                metrics.average_operation_time_ms = (metrics.average_operation_time_ms
                    * (metrics.operations_completed - 1) as f64
                    + execution_time as f64)
                    / metrics.operations_completed as f64;
            }
        }

        Ok(())
    }

    /// Handle validation response
    async fn handle_validation_response(
        &self,
        operation_id: Hash256,
        shard_id: ShardId,
        is_valid: bool,
    ) -> Result<()> {
        tracing::debug!(
            "Received validation response from {:?} for operation {:?}: {}",
            shard_id,
            operation_id,
            is_valid
        );
        // Implementation depends on specific validation logic
        Ok(())
    }

    /// Handle state response
    async fn handle_state_response(
        &self,
        operation_id: Hash256,
        responding_shard: ShardId,
        state_data: HashMap<String, Vec<u8>>,
    ) -> Result<()> {
        tracing::debug!(
            "Received state response from {:?} for operation {:?} with {} keys",
            responding_shard,
            operation_id,
            state_data.len()
        );
        // Implementation depends on specific state query logic
        Ok(())
    }

    /// Handle shard heartbeat
    async fn handle_heartbeat(&self, shard_id: ShardId, load_info: ShardLoadInfo) -> Result<()> {
        tracing::debug!(
            "Received heartbeat from {:?}: TPS={:.2}, CPU={:.1}%",
            shard_id,
            load_info.transactions_per_second,
            load_info.cpu_utilization * 100.0
        );
        // Update shard load information for load balancing decisions
        Ok(())
    }

    /// Cleanup operations that have timed out
    pub async fn cleanup_timed_out_operations(&self) -> Result<usize> {
        let now = Instant::now();
        let mut pending = self.pending_operations.write().await;
        let mut timed_out_ops = Vec::new();

        // Find timed out operations
        for (op_id, operation) in pending.iter() {
            if now > operation.timeout_at {
                timed_out_ops.push(*op_id);
            }
        }

        // Remove timed out operations
        for op_id in &timed_out_ops {
            pending.remove(op_id);
        }

        // Update metrics
        if !timed_out_ops.is_empty() {
            let mut metrics = self.metrics.write().await;
            metrics.operations_timed_out += timed_out_ops.len() as u64;
            metrics.active_operations = pending.len();
        }

        if !timed_out_ops.is_empty() {
            tracing::warn!(
                "Cleaned up {} timed out cross-shard operations",
                timed_out_ops.len()
            );
        }

        Ok(timed_out_ops.len())
    }

    /// Cleanup specific timed out operation
    async fn cleanup_timed_out_operation(&self, operation_id: Hash256) {
        let mut pending = self.pending_operations.write().await;
        pending.remove(&operation_id);

        let mut metrics = self.metrics.write().await;
        metrics.operations_timed_out += 1;
        metrics.active_operations = pending.len();
    }

    /// Get cross-shard bridge metrics
    pub async fn get_metrics(&self) -> CrossShardMetrics {
        let mut metrics = self.metrics.read().await.clone();
        let pending = self.pending_operations.read().await;
        metrics.active_operations = pending.len();
        metrics
    }

    /// Get list of registered shards
    pub async fn get_registered_shards(&self) -> Vec<ShardId> {
        let channels = self.message_channels.read().await;
        channels.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_cross_shard_bridge_creation() {
        let bridge = CrossShardBridge::new("test_bridge".to_string(), Duration::from_secs(30));
        assert_eq!(bridge.bridge_id, "test_bridge");
    }

    #[tokio::test]
    async fn test_shard_registration() {
        let bridge = CrossShardBridge::new("test_bridge".to_string(), Duration::from_secs(30));
        let (sender, _receiver) = mpsc::unbounded_channel();

        bridge.register_shard(ShardId::Consensus(0), sender).await;

        let registered_shards = bridge.get_registered_shards().await;
        assert_eq!(registered_shards.len(), 1);
        assert_eq!(registered_shards[0], ShardId::Consensus(0));
    }

    #[tokio::test]
    async fn test_message_sending() {
        let bridge = CrossShardBridge::new("test_bridge".to_string(), Duration::from_secs(30));
        let (sender, mut receiver) = mpsc::unbounded_channel();

        bridge.register_shard(ShardId::Consensus(0), sender).await;

        let test_message = CrossShardMessage::Heartbeat {
            shard_id: ShardId::Consensus(1),
            timestamp: 12345,
            load_info: ShardLoadInfo {
                transactions_per_second: 1000.0,
                queue_depth: 10,
                cpu_utilization: 0.5,
                memory_usage_mb: 100.0,
            },
        };

        let result = bridge
            .send_message(ShardId::Consensus(0), test_message)
            .await;
        assert!(result.is_ok());

        // Verify message was received
        let received_message = receiver.try_recv();
        assert!(received_message.is_ok());
    }
}
