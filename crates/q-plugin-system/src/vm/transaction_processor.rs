use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Plugin Transaction Processor for Q-NarwhalKnight VM
///
/// This module provides comprehensive transaction processing integration for plugins,
/// enabling them to participate in transaction validation, execution, and state transitions.
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::plugin::{PluginError, PluginHook, PluginId, PluginResult};
use crate::vm::{PluginStateManager, PluginVmBridge};
use q_vm::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction as NarwhalTransaction};
use q_vm::{
    vm::{CallData, ExecutionResult, VmError},
    VirtualMachine,
};

/// Plugin transaction execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginTransactionContext {
    pub plugin_id: PluginId,
    pub transaction_id: String,
    pub sender: u64,
    pub recipient: Option<u64>,
    pub value: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub transaction_type: PluginTransactionType,
    pub execution_phase: TransactionExecutionPhase,
    pub timestamp: u64,
}

/// Types of plugin transactions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginTransactionType {
    /// Standard value transfer with plugin logic
    Transfer { amount: u64, data: Vec<u8> },
    /// Contract deployment through plugin
    ContractDeployment {
        code: Vec<u8>,
        constructor_args: Vec<u8>,
    },
    /// Contract method call
    ContractCall {
        contract_address: u64,
        method: String,
        args: Vec<u8>,
    },
    /// Plugin-specific custom transaction
    PluginSpecific { operation: String, data: Vec<u8> },
    /// Multi-party computation transaction
    MultiPartyComputation {
        participants: Vec<u64>,
        computation_data: Vec<u8>,
    },
    /// State migration transaction
    StateMigration {
        from_state: Vec<u8>,
        to_state: Vec<u8>,
    },
}

/// Transaction execution phases
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionExecutionPhase {
    PreValidation,
    Validation,
    PreExecution,
    Execution,
    PostExecution,
    Finalization,
    Rollback,
}

/// Transaction processing permissions for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPermissions {
    pub can_process_transfers: bool,
    pub can_deploy_contracts: bool,
    pub can_call_contracts: bool,
    pub can_create_custom_transactions: bool,
    pub can_participate_in_mpc: bool,
    pub can_migrate_state: bool,
    pub max_transactions_per_block: u32,
    pub max_gas_per_transaction: u64,
    pub max_value_per_transaction: u64,
}

impl Default for TransactionPermissions {
    fn default() -> Self {
        Self {
            can_process_transfers: true,
            can_deploy_contracts: false,
            can_call_contracts: true,
            can_create_custom_transactions: true,
            can_participate_in_mpc: false,
            can_migrate_state: false,
            max_transactions_per_block: 100,
            max_gas_per_transaction: 1_000_000,
            max_value_per_transaction: u64::MAX,
        }
    }
}

/// Transaction processing metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransactionProcessingMetrics {
    pub transactions_processed: u64,
    pub transactions_successful: u64,
    pub transactions_failed: u64,
    pub total_gas_used: u64,
    pub total_value_processed: u64,
    pub average_execution_time_ms: f64,
    pub contracts_deployed: u64,
    pub contract_calls_made: u64,
    pub custom_operations_executed: u64,
}

/// Plugin transaction processor
pub struct PluginTransactionProcessor {
    /// VM bridge for plugin execution
    vm_bridge: Arc<PluginVmBridge>,

    /// State manager for persistent storage
    state_manager: Arc<PluginStateManager>,

    /// Consensus engine for transaction ordering
    consensus: Arc<NarwhalBullshark>,

    /// Virtual machine for execution
    vm: Arc<VirtualMachine>,

    /// Active transaction contexts
    active_transactions: Arc<RwLock<HashMap<String, PluginTransactionContext>>>,

    /// Plugin transaction permissions
    plugin_permissions: Arc<RwLock<HashMap<PluginId, TransactionPermissions>>>,

    /// Transaction processing metrics
    processing_metrics: Arc<RwLock<HashMap<PluginId, TransactionProcessingMetrics>>>,

    /// Transaction execution locks
    execution_locks: Arc<RwLock<HashMap<String, Arc<Mutex<()>>>>>,
}

impl PluginTransactionProcessor {
    pub fn new(
        vm_bridge: Arc<PluginVmBridge>,
        state_manager: Arc<PluginStateManager>,
        consensus: Arc<NarwhalBullshark>,
        vm: Arc<VirtualMachine>,
    ) -> Self {
        Self {
            vm_bridge,
            state_manager,
            consensus,
            vm,
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            plugin_permissions: Arc::new(RwLock::new(HashMap::new())),
            processing_metrics: Arc::new(RwLock::new(HashMap::new())),
            execution_locks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register plugin for transaction processing
    pub async fn register_plugin_for_transactions(
        &self,
        plugin_id: PluginId,
        permissions: TransactionPermissions,
    ) -> PluginResult<()> {
        // Store permissions
        {
            let mut perms = self.plugin_permissions.write().await;
            perms.insert(plugin_id.clone(), permissions);
        }

        // Initialize metrics
        {
            let mut metrics = self.processing_metrics.write().await;
            metrics.insert(plugin_id, TransactionProcessingMetrics::default());
        }

        Ok(())
    }

    /// Process transaction through plugin
    pub async fn process_plugin_transaction(
        &self,
        plugin_id: &PluginId,
        transaction_type: PluginTransactionType,
        sender: u64,
        recipient: Option<u64>,
        value: u64,
        gas_limit: u64,
        gas_price: u64,
        data: Vec<u8>,
    ) -> PluginResult<ExecutionResult> {
        let transaction_id = uuid::Uuid::new_v4().to_string();

        // Create transaction context
        let context = PluginTransactionContext {
            plugin_id: plugin_id.clone(),
            transaction_id: transaction_id.clone(),
            sender,
            recipient,
            value,
            gas_limit,
            gas_price,
            transaction_type: transaction_type.clone(),
            execution_phase: TransactionExecutionPhase::PreValidation,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Acquire execution lock
        let lock_arc = self.acquire_execution_lock(&transaction_id).await;
        let _lock = lock_arc.lock().await;

        // Store active transaction
        {
            let mut active = self.active_transactions.write().await;
            active.insert(transaction_id.clone(), context.clone());
        }

        let start_time = std::time::Instant::now();
        let mut final_result = ExecutionResult {
            success: false,
            return_data: Vec::new(),
            gas_used: 0,
            logs: Vec::new(),
            error: Some("Transaction not executed".to_string()),
        };

        // Execute transaction through phases
        match self
            .execute_transaction_phases(plugin_id, &transaction_id, &transaction_type, &data)
            .await
        {
            Ok(result) => {
                final_result = result;
            }
            Err(e) => {
                // Execute rollback phase
                self.execute_transaction_phase(
                    plugin_id,
                    &transaction_id,
                    TransactionExecutionPhase::Rollback,
                    &data,
                )
                .await
                .ok(); // Ignore rollback errors

                final_result.error = Some(e.to_string());
            }
        }

        // Update metrics
        let execution_time = start_time.elapsed();
        self.update_transaction_metrics(plugin_id, &final_result, execution_time)
            .await;

        // Clean up
        {
            let mut active = self.active_transactions.write().await;
            active.remove(&transaction_id);
        }
        {
            let mut locks = self.execution_locks.write().await;
            locks.remove(&transaction_id);
        }

        Ok(final_result)
    }

    /// Get transaction processing metrics
    pub async fn get_transaction_metrics(
        &self,
        plugin_id: &PluginId,
    ) -> Option<TransactionProcessingMetrics> {
        let metrics = self.processing_metrics.read().await;
        metrics.get(plugin_id).cloned()
    }

    /// Get active transactions for plugin
    pub async fn get_active_transactions(
        &self,
        plugin_id: &PluginId,
    ) -> Vec<PluginTransactionContext> {
        let active = self.active_transactions.read().await;
        active
            .values()
            .filter(|ctx| &ctx.plugin_id == plugin_id)
            .cloned()
            .collect()
    }

    // Private helper methods

    async fn execute_transaction_phases(
        &self,
        plugin_id: &PluginId,
        transaction_id: &str,
        _transaction_type: &PluginTransactionType,
        data: &[u8],
    ) -> PluginResult<ExecutionResult> {
        let phases = vec![
            TransactionExecutionPhase::PreValidation,
            TransactionExecutionPhase::Validation,
            TransactionExecutionPhase::PreExecution,
            TransactionExecutionPhase::Execution,
            TransactionExecutionPhase::PostExecution,
            TransactionExecutionPhase::Finalization,
        ];

        let mut execution_result = ExecutionResult {
            success: true,
            return_data: Vec::new(),
            gas_used: 0,
            logs: Vec::new(),
            error: None,
        };

        for phase in phases {
            // Execute phase
            match self
                .execute_transaction_phase(plugin_id, transaction_id, phase.clone(), data)
                .await
            {
                Ok(phase_result) => {
                    // Accumulate results
                    execution_result.gas_used += phase_result.gas_used;
                    execution_result.logs.extend(phase_result.logs);

                    // Use the execution phase result as the final return data
                    if phase == TransactionExecutionPhase::Execution {
                        execution_result.return_data = phase_result.return_data;
                    }

                    if !phase_result.success {
                        execution_result.success = false;
                        execution_result.error = phase_result.error;
                        return Err(PluginError::ExecutionFailed(
                            execution_result
                                .error
                                .clone()
                                .unwrap_or("Phase execution failed".to_string()),
                        ));
                    }
                }
                Err(e) => {
                    execution_result.success = false;
                    execution_result.error = Some(e.to_string());
                    return Err(e);
                }
            }
        }

        Ok(execution_result)
    }

    async fn execute_transaction_phase(
        &self,
        plugin_id: &PluginId,
        transaction_id: &str,
        phase: TransactionExecutionPhase,
        data: &[u8],
    ) -> PluginResult<ExecutionResult> {
        // Determine hook based on phase
        let hook = match phase {
            TransactionExecutionPhase::PreValidation => {
                PluginHook::Custom("pre_validate_transaction".to_string())
            }
            TransactionExecutionPhase::Validation => {
                PluginHook::Custom("validate_transaction".to_string())
            }
            TransactionExecutionPhase::PreExecution => PluginHook::BeforeTransaction,
            TransactionExecutionPhase::Execution => {
                PluginHook::Custom("execute_transaction".to_string())
            }
            TransactionExecutionPhase::PostExecution => PluginHook::AfterTransaction,
            TransactionExecutionPhase::Finalization => {
                PluginHook::Custom("finalize_transaction".to_string())
            }
            TransactionExecutionPhase::Rollback => {
                PluginHook::Custom("rollback_transaction".to_string())
            }
        };

        // Create call data for this phase
        let call_data = self
            .create_transaction_call_data(transaction_id, &phase)
            .await?;

        // Execute through VM bridge
        match self
            .vm_bridge
            .execute_plugin_in_vm(plugin_id, hook, data, Some(call_data))
            .await
        {
            Ok(result_data) => Ok(ExecutionResult {
                success: true,
                return_data: result_data,
                gas_used: 10_000, // Estimated gas for transaction phase
                logs: vec![format!(
                    "Executed {:?} phase for transaction {}",
                    phase, transaction_id
                )],
                error: None,
            }),
            Err(e) => Err(e),
        }
    }

    async fn create_transaction_call_data(
        &self,
        transaction_id: &str,
        phase: &TransactionExecutionPhase,
    ) -> PluginResult<CallData> {
        let active = self.active_transactions.read().await;
        let context = active
            .get(transaction_id)
            .ok_or_else(|| PluginError::NotFound(format!("Transaction {}", transaction_id)))?;

        Ok(CallData {
            contract_address: 0, // Special address for transaction processing
            function: format!("transaction_{:?}", phase).to_lowercase(),
            arguments: bincode::serialize(context)
                .map_err(|e| PluginError::Internal(format!("Serialization failed: {}", e)))?,
            sender: context.sender,
            gas_limit: context.gas_limit,
            gas_price: context.gas_price,
            value: context.value,
            is_rwa_operation: false,
            bulk_operation_count: 0,
        })
    }

    async fn update_transaction_metrics(
        &self,
        plugin_id: &PluginId,
        result: &ExecutionResult,
        execution_time: std::time::Duration,
    ) {
        let mut metrics = self.processing_metrics.write().await;
        if let Some(plugin_metrics) = metrics.get_mut(plugin_id) {
            plugin_metrics.transactions_processed += 1;

            if result.success {
                plugin_metrics.transactions_successful += 1;
            } else {
                plugin_metrics.transactions_failed += 1;
            }

            plugin_metrics.total_gas_used += result.gas_used;

            // Update average execution time
            let total_time = plugin_metrics.average_execution_time_ms
                * (plugin_metrics.transactions_processed - 1) as f64;
            plugin_metrics.average_execution_time_ms = (total_time
                + execution_time.as_millis() as f64)
                / plugin_metrics.transactions_processed as f64;
        }
    }

    async fn acquire_execution_lock(&self, transaction_id: &str) -> Arc<Mutex<()>> {
        // Get or create lock for this transaction
        let lock_arc = {
            let mut locks = self.execution_locks.write().await;
            locks
                .entry(transaction_id.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        lock_arc
    }
}
