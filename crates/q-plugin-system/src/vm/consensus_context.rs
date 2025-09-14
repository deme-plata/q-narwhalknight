use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Plugin Consensus Context for Q-NarwhalKnight Integration
///
/// This module provides execution context for plugins within the Q-NarwhalKnight consensus environment,
/// enabling plugins to participate in consensus operations, block processing, and state validation.
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::plugin::{PluginError, PluginHook, PluginId, PluginResult};
use crate::vm::PluginVmBridge;
use q_vm::consensus::narwhal_bullshark::NarwhalBullshark;
use q_vm::vm::{CallData, ExecutionResult};

/// Consensus execution context for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConsensusContext {
    pub plugin_id: PluginId,
    pub consensus_round: u64,
    pub block_height: u64,
    pub block_hash: Option<String>,
    pub timestamp: u64,
    pub validator_id: String,
    pub consensus_permissions: ConsensusPermissions,
    pub execution_phase: ConsensusExecutionPhase,
}

/// Permissions for plugin consensus operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusPermissions {
    pub can_propose_blocks: bool,
    pub can_validate_transactions: bool,
    pub can_participate_in_voting: bool,
    pub can_access_network_state: bool,
    pub can_read_consensus_data: bool,
    pub can_submit_evidence: bool,
    pub max_consensus_operations_per_round: u32,
}

impl Default for ConsensusPermissions {
    fn default() -> Self {
        Self {
            can_propose_blocks: false,
            can_validate_transactions: true,
            can_participate_in_voting: false,
            can_access_network_state: true,
            can_read_consensus_data: true,
            can_submit_evidence: false,
            max_consensus_operations_per_round: 10,
        }
    }
}

/// Phase of consensus execution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsensusExecutionPhase {
    PreConsensus,
    TransactionValidation,
    BlockProposal,
    BlockValidation,
    Voting,
    PostConsensus,
    Finalization,
}

/// Consensus operation metrics for plugins
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusOperationMetrics {
    pub operations_performed: u64,
    pub blocks_proposed: u64,
    pub transactions_validated: u64,
    pub votes_cast: u64,
    pub consensus_rounds_participated: u64,
    pub average_operation_time_ms: f64,
    pub errors_encountered: u64,
}

/// Plugin consensus context manager
pub struct PluginConsensusContextManager {
    /// Active consensus contexts by plugin
    active_contexts: Arc<RwLock<HashMap<PluginId, PluginConsensusContext>>>,

    /// Consensus operation metrics
    operation_metrics: Arc<RwLock<HashMap<PluginId, ConsensusOperationMetrics>>>,

    /// Plugin VM bridge for integration
    vm_bridge: Arc<PluginVmBridge>,

    /// NarwhalBullshark consensus engine
    consensus: Arc<NarwhalBullshark>,

    /// Operation locks for thread safety
    operation_locks: Arc<RwLock<HashMap<PluginId, Arc<Mutex<()>>>>>,

    /// Consensus hooks registry
    consensus_hooks: Arc<RwLock<HashMap<ConsensusExecutionPhase, Vec<PluginId>>>>,
}

impl PluginConsensusContextManager {
    pub fn new(vm_bridge: Arc<PluginVmBridge>, consensus: Arc<NarwhalBullshark>) -> Self {
        Self {
            active_contexts: Arc::new(RwLock::new(HashMap::new())),
            operation_metrics: Arc::new(RwLock::new(HashMap::new())),
            vm_bridge,
            consensus,
            operation_locks: Arc::new(RwLock::new(HashMap::new())),
            consensus_hooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register plugin for consensus participation
    pub async fn register_plugin_for_consensus(
        &self,
        plugin_id: PluginId,
        permissions: ConsensusPermissions,
        validator_id: String,
    ) -> PluginResult<()> {
        let context = PluginConsensusContext {
            plugin_id: plugin_id.clone(),
            consensus_round: 0,
            block_height: 0,
            block_hash: None,
            timestamp: chrono::Utc::now().timestamp() as u64,
            validator_id,
            consensus_permissions: permissions,
            execution_phase: ConsensusExecutionPhase::PreConsensus,
        };

        // Store context
        {
            let mut contexts = self.active_contexts.write().await;
            contexts.insert(plugin_id.clone(), context);
        }

        // Initialize metrics
        {
            let mut metrics = self.operation_metrics.write().await;
            metrics.insert(plugin_id.clone(), ConsensusOperationMetrics::default());
        }

        // Create operation lock
        {
            let mut locks = self.operation_locks.write().await;
            locks.insert(plugin_id, Arc::new(Mutex::new(())));
        }

        Ok(())
    }

    /// Execute plugin in specific consensus phase
    pub async fn execute_plugin_in_consensus_phase(
        &self,
        plugin_id: &PluginId,
        phase: ConsensusExecutionPhase,
        hook: PluginHook,
        data: &[u8],
    ) -> PluginResult<Vec<u8>> {
        // Acquire operation lock
        let lock_arc = self.acquire_operation_lock(plugin_id).await?;
        let _lock = lock_arc.lock().await;

        // Update context phase
        self.update_consensus_phase(plugin_id, phase.clone())
            .await?;

        // Check permissions for this phase
        self.check_phase_permissions(plugin_id, &phase).await?;

        let start_time = std::time::Instant::now();

        // Execute plugin with consensus context
        let result = self
            .vm_bridge
            .execute_plugin_in_vm(
                plugin_id,
                hook,
                data,
                Some(self.create_consensus_call_data(plugin_id, &phase).await?),
            )
            .await;

        // Update metrics
        let execution_time = start_time.elapsed();
        self.update_operation_metrics(plugin_id, &phase, execution_time, &result)
            .await;

        result
    }

    /// Process block through plugins in sequence
    pub async fn process_block_with_plugins(
        &self,
        block_data: &[u8],
        block_height: u64,
        block_hash: String,
    ) -> PluginResult<Vec<ExecutionResult>> {
        let mut results = Vec::new();

        // Get all plugins registered for block processing
        let plugin_ids = self
            .get_plugins_for_phase(ConsensusExecutionPhase::BlockValidation)
            .await;

        for plugin_id in plugin_ids {
            // Update context for block processing
            {
                let mut contexts = self.active_contexts.write().await;
                if let Some(context) = contexts.get_mut(&plugin_id) {
                    context.block_height = block_height;
                    context.block_hash = Some(block_hash.clone());
                    context.timestamp = chrono::Utc::now().timestamp() as u64;
                }
            }

            // Execute block validation hook
            match self
                .execute_plugin_in_consensus_phase(
                    &plugin_id,
                    ConsensusExecutionPhase::BlockValidation,
                    PluginHook::BeforeBlock,
                    block_data,
                )
                .await
            {
                Ok(result_data) => {
                    results.push(ExecutionResult {
                        success: true,
                        return_data: result_data,
                        gas_used: 1000, // Estimated gas for consensus operations
                        logs: vec![format!(
                            "Plugin {:?} validated block {}",
                            plugin_id, block_height
                        )],
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(ExecutionResult {
                        success: false,
                        return_data: Vec::new(),
                        gas_used: 0,
                        logs: vec![format!("Plugin {:?} failed block validation", plugin_id)],
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Register plugin hook for specific consensus phase
    pub async fn register_consensus_hook(
        &self,
        plugin_id: PluginId,
        phase: ConsensusExecutionPhase,
    ) -> PluginResult<()> {
        let mut hooks = self.consensus_hooks.write().await;
        let phase_hooks = hooks.entry(phase).or_insert_with(Vec::new);

        if !phase_hooks.contains(&plugin_id) {
            phase_hooks.push(plugin_id);
        }

        Ok(())
    }

    /// Get consensus metrics for plugin
    pub async fn get_consensus_metrics(
        &self,
        plugin_id: &PluginId,
    ) -> Option<ConsensusOperationMetrics> {
        let metrics = self.operation_metrics.read().await;
        metrics.get(plugin_id).cloned()
    }

    /// Get all active consensus contexts
    pub async fn list_active_consensus_contexts(&self) -> Vec<(PluginId, PluginConsensusContext)> {
        let contexts = self.active_contexts.read().await;
        contexts
            .iter()
            .map(|(id, ctx)| (id.clone(), ctx.clone()))
            .collect()
    }

    // Private helper methods

    async fn acquire_operation_lock(&self, plugin_id: &PluginId) -> PluginResult<Arc<Mutex<()>>> {
        let locks = self.operation_locks.read().await;
        locks
            .get(plugin_id)
            .cloned()
            .ok_or_else(|| PluginError::NotFound(format!("Operation lock for {:?}", plugin_id)))
    }

    async fn update_consensus_phase(
        &self,
        plugin_id: &PluginId,
        phase: ConsensusExecutionPhase,
    ) -> PluginResult<()> {
        let mut contexts = self.active_contexts.write().await;
        if let Some(context) = contexts.get_mut(plugin_id) {
            context.execution_phase = phase;
            Ok(())
        } else {
            Err(PluginError::NotFound(format!(
                "Consensus context for {:?}",
                plugin_id
            )))
        }
    }

    async fn check_phase_permissions(
        &self,
        plugin_id: &PluginId,
        phase: &ConsensusExecutionPhase,
    ) -> PluginResult<()> {
        let contexts = self.active_contexts.read().await;
        let context = contexts.get(plugin_id).ok_or_else(|| {
            PluginError::NotFound(format!("Consensus context for {:?}", plugin_id))
        })?;

        match phase {
            ConsensusExecutionPhase::BlockProposal => {
                if !context.consensus_permissions.can_propose_blocks {
                    return Err(PluginError::PermissionDenied(
                        "Block proposal not allowed".to_string(),
                    ));
                }
            }
            ConsensusExecutionPhase::TransactionValidation => {
                if !context.consensus_permissions.can_validate_transactions {
                    return Err(PluginError::PermissionDenied(
                        "Transaction validation not allowed".to_string(),
                    ));
                }
            }
            ConsensusExecutionPhase::Voting => {
                if !context.consensus_permissions.can_participate_in_voting {
                    return Err(PluginError::PermissionDenied(
                        "Voting not allowed".to_string(),
                    ));
                }
            }
            _ => {} // Other phases are generally allowed
        }

        Ok(())
    }

    async fn create_consensus_call_data(
        &self,
        plugin_id: &PluginId,
        phase: &ConsensusExecutionPhase,
    ) -> PluginResult<CallData> {
        let context = self.get_consensus_context(plugin_id).await.ok_or_else(|| {
            PluginError::NotFound(format!("Consensus context for {:?}", plugin_id))
        })?;

        Ok(CallData {
            contract_address: 0, // Special address for consensus operations
            function: format!("consensus_{:?}", phase).to_lowercase(),
            arguments: bincode::serialize(&context)
                .map_err(|e| PluginError::Internal(format!("Serialization failed: {}", e)))?,
            sender: 0,          // Consensus system
            gas_limit: 100_000, // Higher gas limit for consensus operations
            gas_price: 0,       // Consensus operations are free
            value: 0,
            is_rwa_operation: false,
            bulk_operation_count: 0,
        })
    }

    async fn get_consensus_context(&self, plugin_id: &PluginId) -> Option<PluginConsensusContext> {
        let contexts = self.active_contexts.read().await;
        contexts.get(plugin_id).cloned()
    }

    async fn update_operation_metrics(
        &self,
        plugin_id: &PluginId,
        phase: &ConsensusExecutionPhase,
        execution_time: std::time::Duration,
        result: &PluginResult<Vec<u8>>,
    ) {
        let mut metrics = self.operation_metrics.write().await;
        if let Some(plugin_metrics) = metrics.get_mut(plugin_id) {
            plugin_metrics.operations_performed += 1;

            match phase {
                ConsensusExecutionPhase::BlockProposal => plugin_metrics.blocks_proposed += 1,
                ConsensusExecutionPhase::TransactionValidation => {
                    plugin_metrics.transactions_validated += 1
                }
                ConsensusExecutionPhase::Voting => plugin_metrics.votes_cast += 1,
                _ => {}
            }

            // Update average operation time
            let total_time = plugin_metrics.average_operation_time_ms
                * (plugin_metrics.operations_performed - 1) as f64;
            plugin_metrics.average_operation_time_ms = (total_time
                + execution_time.as_millis() as f64)
                / plugin_metrics.operations_performed as f64;

            if result.is_err() {
                plugin_metrics.errors_encountered += 1;
            }
        }
    }

    async fn get_plugins_for_phase(&self, phase: ConsensusExecutionPhase) -> Vec<PluginId> {
        let hooks = self.consensus_hooks.read().await;
        hooks.get(&phase).cloned().unwrap_or_default()
    }
}
