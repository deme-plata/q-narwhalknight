use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Plugin-VM Bridge Integration for Q-NarwhalKnight
///
/// This module provides deep integration between the plugin system and Q-VM,
/// enabling plugins to interact with consensus, state management, and transaction processing.
/// Adapted from Orobit Chimeras for Q-NarwhalKnight architecture.
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::plugin::{PluginError, PluginHook, PluginId, PluginManager, PluginResult, PluginState};
use crate::vm::PluginStateManager;

// Q-VM integration imports
use q_vm::consensus::narwhal_bullshark::NarwhalBullshark;
use q_vm::{
    vm::{CallData, ContractState, ExecutionResult, StateAccess, VmError},
    VirtualMachine,
};

/// Plugin execution environment within the VM context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginVmContext {
    pub plugin_id: PluginId,
    pub vm_state_access: bool,
    pub consensus_access: bool,
    pub transaction_processing: bool,
    pub resource_limits: PluginResourceLimits,
    pub execution_timestamp: u64,
    pub block_height: u64,
}

/// Resource limits for plugin execution within VM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResourceLimits {
    pub max_gas_per_call: u64,
    pub max_state_reads: u32,
    pub max_state_writes: u32,
    pub max_consensus_operations: u32,
    pub max_execution_time_ms: u64,
    pub max_memory_bytes: u64,
}

impl Default for PluginResourceLimits {
    fn default() -> Self {
        Self {
            max_gas_per_call: 1_000_000,   // 1M gas limit
            max_state_reads: 100,          // 100 state reads per call
            max_state_writes: 50,          // 50 state writes per call
            max_consensus_operations: 10,  // 10 consensus ops per call
            max_execution_time_ms: 5000,   // 5 second timeout
            max_memory_bytes: 100_000_000, // 100MB memory limit
        }
    }
}

/// Plugin execution metrics within VM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginVmMetrics {
    pub gas_consumed: u64,
    pub state_reads: u32,
    pub state_writes: u32,
    pub consensus_operations: u32,
    pub execution_time_ms: u64,
    pub memory_usage_bytes: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
}

impl Default for PluginVmMetrics {
    fn default() -> Self {
        Self {
            gas_consumed: 0,
            state_reads: 0,
            state_writes: 0,
            consensus_operations: 0,
            execution_time_ms: 0,
            memory_usage_bytes: 0,
            successful_calls: 0,
            failed_calls: 0,
        }
    }
}

/// Main bridge between plugin system and Q-NarwhalKnight VM
pub struct PluginVmBridge {
    vm: Arc<VirtualMachine>,
    consensus: Arc<NarwhalBullshark>,
    plugin_manager: Arc<PluginManager>,
    plugin_contexts: Arc<RwLock<HashMap<PluginId, PluginVmContext>>>,
    plugin_metrics: Arc<RwLock<HashMap<PluginId, PluginVmMetrics>>>,
    state_access: Arc<dyn StateAccess>,
    state_manager: Arc<PluginStateManager>,
}

impl PluginVmBridge {
    pub fn new(
        vm: Arc<VirtualMachine>,
        consensus: Arc<NarwhalBullshark>,
        plugin_manager: Arc<PluginManager>,
        state_access: Arc<dyn StateAccess>,
        state_manager: Arc<PluginStateManager>,
    ) -> Self {
        Self {
            vm,
            consensus,
            plugin_manager,
            plugin_contexts: Arc::new(RwLock::new(HashMap::new())),
            plugin_metrics: Arc::new(RwLock::new(HashMap::new())),
            state_access,
            state_manager,
        }
    }

    /// Register a plugin for VM integration
    pub async fn register_plugin_for_vm(
        &self,
        plugin_id: PluginId,
        vm_permissions: VmPermissions,
    ) -> Result<(), PluginError> {
        let context = PluginVmContext {
            plugin_id: plugin_id.clone(),
            vm_state_access: vm_permissions.state_access,
            consensus_access: vm_permissions.consensus_access,
            transaction_processing: vm_permissions.transaction_processing,
            resource_limits: vm_permissions.resource_limits.unwrap_or_default(),
            execution_timestamp: chrono::Utc::now().timestamp() as u64,
            block_height: self.get_current_block_height().await.unwrap_or(0),
        };

        let mut contexts = self.plugin_contexts.write().await;
        contexts.insert(plugin_id.clone(), context);

        let mut metrics = self.plugin_metrics.write().await;
        metrics.insert(plugin_id, PluginVmMetrics::default());

        Ok(())
    }

    /// Execute a plugin hook within the VM context
    pub async fn execute_plugin_in_vm(
        &self,
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
        call_data: Option<CallData>,
    ) -> PluginResult<Vec<u8>> {
        let context = {
            let contexts = self.plugin_contexts.read().await;
            contexts
                .get(plugin_id)
                .cloned()
                .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?
        };

        // Check resource limits before execution
        self.check_resource_limits(&context).await?;

        let start_time = std::time::Instant::now();

        // Create enhanced execution context with VM access
        let vm_execution_context = PluginVmExecutionContext {
            plugin_context: context.clone(),
            vm_bridge: Arc::new(self.clone()),
            call_data,
        };

        // Execute the plugin with VM context
        let result = self
            .plugin_manager
            .execute_hook_with_context(plugin_id, hook, data, vm_execution_context)
            .await;

        // Update metrics
        let execution_time = start_time.elapsed().as_millis() as u64;
        self.update_plugin_metrics(plugin_id, execution_time, &result)
            .await;

        result
    }

    /// Execute a plugin as part of transaction processing
    pub async fn execute_plugin_transaction(
        &self,
        plugin_id: &PluginId,
        transaction_data: Vec<u8>,
        sender: u64,
        gas_limit: u64,
    ) -> Result<ExecutionResult, VmError> {
        let call_data = CallData {
            contract_address: 0, // Plugin contracts use special address space
            function: "plugin_transaction".to_string(),
            arguments: transaction_data,
            sender,
            gas_limit,
            gas_price: 1,
            value: 0,
            is_rwa_operation: false,
            bulk_operation_count: 0,
        };

        // Execute plugin in VM context
        let arguments = call_data.arguments.clone();
        let plugin_result = self
            .execute_plugin_in_vm(
                plugin_id,
                PluginHook::Custom("transaction_processing".to_string()),
                &arguments,
                Some(call_data),
            )
            .await
            .map_err(|e| VmError::ExecutionError(e.to_string()))?;

        // Convert plugin result to VM execution result
        Ok(ExecutionResult {
            success: true,
            return_data: plugin_result,
            gas_used: gas_limit / 10, // Plugins use optimized gas consumption
            logs: vec![format!("Plugin {} executed transaction", plugin_id)],
            error: None,
        })
    }

    /// Allow plugins to interact with consensus
    pub async fn plugin_consensus_operation(
        &self,
        plugin_id: &PluginId,
        operation: ConsensusOperation,
    ) -> PluginResult<ConsensusResult> {
        let context = {
            let contexts = self.plugin_contexts.read().await;
            contexts
                .get(plugin_id)
                .cloned()
                .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?
        };

        if !context.consensus_access {
            return Err(PluginError::PermissionDenied(
                "Plugin does not have consensus access permission".to_string(),
            ));
        }

        match operation {
            ConsensusOperation::ProposeBlock(block_data) => {
                // Interact with Narwhal consensus to propose a block
                let result = self
                    .consensus
                    .propose_block(block_data)
                    .await
                    .map_err(|e| PluginError::ExecutionFailed(e.to_string()))?;

                Ok(ConsensusResult::BlockProposed(result))
            }
            ConsensusOperation::ValidateTransaction(tx_data) => {
                // Validate transaction through consensus
                let is_valid = self
                    .consensus
                    .validate_transaction(&tx_data)
                    .await
                    .map_err(|e| PluginError::ExecutionFailed(e.to_string()))?;

                Ok(ConsensusResult::TransactionValidated(is_valid))
            }
            ConsensusOperation::GetNetworkState => {
                // Get current network consensus state
                let state = self
                    .consensus
                    .get_network_state()
                    .await
                    .map_err(|e| PluginError::ExecutionFailed(e.to_string()))?;

                Ok(ConsensusResult::NetworkState(state))
            }
        }
    }

    /// Get plugin execution metrics
    pub async fn get_plugin_metrics(&self, plugin_id: &PluginId) -> Option<PluginVmMetrics> {
        let metrics = self.plugin_metrics.read().await;
        metrics.get(plugin_id).cloned()
    }

    /// Get all registered plugin contexts
    pub async fn list_plugin_contexts(&self) -> Vec<(PluginId, PluginVmContext)> {
        let contexts = self.plugin_contexts.read().await;
        contexts
            .iter()
            .map(|(id, ctx)| (id.clone(), ctx.clone()))
            .collect()
    }

    /// Initialize plugin state in VM with full state management
    pub async fn initialize_plugin_vm_state(
        &self,
        plugin_id: PluginId,
        initial_state: PluginState,
        vm_permissions: VmPermissions,
    ) -> PluginResult<()> {
        let context = PluginVmContext {
            plugin_id: plugin_id.clone(),
            vm_state_access: vm_permissions.state_access,
            consensus_access: vm_permissions.consensus_access,
            transaction_processing: vm_permissions.transaction_processing,
            resource_limits: vm_permissions.resource_limits.clone().unwrap_or_default(),
            execution_timestamp: chrono::Utc::now().timestamp() as u64,
            block_height: self.get_current_block_height().await.unwrap_or(0),
        };

        // Initialize in state manager with persistence
        self.state_manager
            .initialize_plugin_state(plugin_id.clone(), initial_state, context.clone())
            .await?;

        // Also register in bridge for runtime access
        self.register_plugin_for_vm(plugin_id, vm_permissions)
            .await?;

        Ok(())
    }

    /// Update plugin state atomically through state manager
    pub async fn update_plugin_vm_state<F, R>(
        &self,
        plugin_id: &PluginId,
        update_fn: F,
    ) -> PluginResult<R>
    where
        F: FnOnce(&mut PluginState) -> PluginResult<R> + Send,
    {
        self.state_manager
            .update_plugin_state(plugin_id, update_fn)
            .await
    }

    /// Get plugin state through state manager
    pub async fn get_plugin_vm_state(
        &self,
        plugin_id: &PluginId,
    ) -> Option<Arc<tokio::sync::Mutex<PluginState>>> {
        self.state_manager.get_plugin_state(plugin_id).await
    }

    /// Remove plugin and its state completely
    pub async fn remove_plugin_vm_state(&self, plugin_id: &PluginId) -> PluginResult<()> {
        // Remove from state manager
        self.state_manager.remove_plugin_state(plugin_id).await?;

        // Remove from bridge contexts
        {
            let mut contexts = self.plugin_contexts.write().await;
            contexts.remove(plugin_id);
        }

        {
            let mut metrics = self.plugin_metrics.write().await;
            metrics.remove(plugin_id);
        }

        Ok(())
    }

    // Private helper methods

    async fn check_resource_limits(&self, context: &PluginVmContext) -> PluginResult<()> {
        let metrics = {
            let metrics_map = self.plugin_metrics.read().await;
            metrics_map
                .get(&context.plugin_id)
                .cloned()
                .unwrap_or_default()
        };

        if metrics.gas_consumed >= context.resource_limits.max_gas_per_call {
            return Err(PluginError::ResourceLimitExceeded(
                "Gas limit exceeded".to_string(),
            ));
        }

        if metrics.memory_usage_bytes >= context.resource_limits.max_memory_bytes {
            return Err(PluginError::ResourceLimitExceeded(
                "Memory limit exceeded".to_string(),
            ));
        }

        Ok(())
    }

    async fn update_plugin_metrics(
        &self,
        plugin_id: &PluginId,
        execution_time: u64,
        result: &PluginResult<Vec<u8>>,
    ) {
        let mut metrics_map = self.plugin_metrics.write().await;
        if let Some(metrics) = metrics_map.get_mut(plugin_id) {
            metrics.execution_time_ms += execution_time;

            match result {
                Ok(_) => metrics.successful_calls += 1,
                Err(_) => metrics.failed_calls += 1,
            }

            // Estimate resource usage (this could be more sophisticated)
            metrics.gas_consumed += execution_time * 100; // Rough gas estimation
            metrics.memory_usage_bytes += 1024; // Rough memory estimation
        }
    }

    async fn get_current_block_height(&self) -> Result<u64, VmError> {
        // This would interact with consensus to get current height
        // For now, return a placeholder
        Ok(0)
    }
}

impl Clone for PluginVmBridge {
    fn clone(&self) -> Self {
        Self {
            vm: self.vm.clone(),
            consensus: self.consensus.clone(),
            plugin_manager: self.plugin_manager.clone(),
            plugin_contexts: self.plugin_contexts.clone(),
            plugin_metrics: self.plugin_metrics.clone(),
            state_access: self.state_access.clone(),
            state_manager: self.state_manager.clone(),
        }
    }
}

/// VM permissions for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmPermissions {
    pub state_access: bool,
    pub consensus_access: bool,
    pub transaction_processing: bool,
    pub resource_limits: Option<PluginResourceLimits>,
}

/// Enhanced plugin execution context with VM access
pub struct PluginVmExecutionContext {
    pub plugin_context: PluginVmContext,
    pub vm_bridge: Arc<PluginVmBridge>,
    pub call_data: Option<CallData>,
}

/// Consensus operations that plugins can perform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusOperation {
    ProposeBlock(Vec<u8>),
    ValidateTransaction(Vec<u8>),
    GetNetworkState,
}

/// Results from consensus operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusResult {
    BlockProposed(String),
    TransactionValidated(bool),
    NetworkState(String),
}

/// State access implementation for plugins
pub struct PluginStateAccess {
    vm_bridge: Arc<PluginVmBridge>,
    plugin_id: PluginId,
}

impl PluginStateAccess {
    pub fn new(vm_bridge: Arc<PluginVmBridge>, plugin_id: PluginId) -> Self {
        Self {
            vm_bridge,
            plugin_id,
        }
    }
}

#[async_trait]
impl StateAccess for PluginStateAccess {
    async fn get_contract(&self, address: u64) -> Result<Option<Vec<u8>>, VmError> {
        // Check if plugin has permission for state access
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        // Delegate to underlying state access
        self.vm_bridge.state_access.get_contract(address).await
    }

    async fn get_storage(&self, address: u64, key: &[u8]) -> Result<Option<Vec<u8>>, VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        // Update metrics
        {
            let mut metrics_map = self.vm_bridge.plugin_metrics.write().await;
            if let Some(metrics) = metrics_map.get_mut(&self.plugin_id) {
                metrics.state_reads += 1;
            }
        }

        self.vm_bridge.state_access.get_storage(address, key).await
    }

    async fn set_storage(&self, address: u64, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        // Update metrics
        {
            let mut metrics_map = self.vm_bridge.plugin_metrics.write().await;
            if let Some(metrics) = metrics_map.get_mut(&self.plugin_id) {
                metrics.state_writes += 1;
            }
        }

        self.vm_bridge
            .state_access
            .set_storage(address, key, value)
            .await
    }

    async fn get_balance(&self, address: u64) -> Result<u64, VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        self.vm_bridge.state_access.get_balance(address).await
    }

    async fn set_balance(&self, address: u64, amount: u64) -> Result<(), VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        self.vm_bridge
            .state_access
            .set_balance(address, amount)
            .await
    }

    async fn get_nonce(&self, address: u64) -> Result<u64, VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        self.vm_bridge.state_access.get_nonce(address).await
    }

    async fn get_contract_state(&self, address: u64) -> Result<Option<ContractState>, VmError> {
        let contexts = self.vm_bridge.plugin_contexts.read().await;
        let context = contexts
            .get(&self.plugin_id)
            .ok_or_else(|| VmError::ExecutionError("Plugin not found".to_string()))?;

        if !context.vm_state_access {
            return Err(VmError::ExecutionError(
                "Plugin lacks state access permission".to_string(),
            ));
        }

        self.vm_bridge
            .state_access
            .get_contract_state(address)
            .await
    }
}
