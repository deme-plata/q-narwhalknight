/// Q-NarwhalKnight Plugin System
///
/// Comprehensive plugin architecture for the quantum consensus VM
/// Adapted from Orobit Chimeras plugin system with Q-NarwhalKnight integration
///
/// ## Modules
///
/// - **plugin**: Core plugin types, management, and lifecycle
/// - **vm**: VM integration, state management, and consensus context
/// - **network**: P2P distribution protocol for plugin sharing
/// - **persistence**: RocksDB-backed decentralized storage with consensus verification
///
/// ## Decentralized Plugin Architecture
///
/// Every node in the Q-NarwhalKnight network:
/// 1. **Stores** plugins in local RocksDB with hash verification
/// 2. **Verifies** plugin state using Ed25519/Dilithium5 signatures
/// 3. **Replicates** via P2P gossipsub to all peers
/// 4. **Achieves consensus** through DAG-Knight inclusion
///
/// This ensures truly decentralized plugin execution where every node
/// independently verifies and stores plugin state.
pub mod network;
pub mod persistence;
pub mod plugin;
pub mod vm;

pub use network::*;
pub use persistence::*;
pub use plugin::*;
pub use vm::*;

use q_vm::vm::{ExecutionResult, StateAccess};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Plugin system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSystemConfig {
    pub max_plugins: u32,
    pub max_memory_per_plugin: u64,
    pub max_execution_time_ms: u64,
    pub enable_consensus_integration: bool,
    pub enable_transaction_processing: bool,
    pub enable_state_persistence: bool,
}

impl Default for PluginSystemConfig {
    fn default() -> Self {
        Self {
            max_plugins: 100,
            max_memory_per_plugin: 100_000_000, // 100MB
            max_execution_time_ms: 5000,        // 5 seconds
            enable_consensus_integration: true,
            enable_transaction_processing: true,
            enable_state_persistence: true,
        }
    }
}

/// Main plugin system coordinator
pub struct PluginSystem {
    config: PluginSystemConfig,
    plugin_manager: Arc<PluginManager>,
    vm_bridge: Arc<PluginVmBridge>,
    state_manager: Arc<PluginStateManager>,
    consensus_context_manager: Arc<PluginConsensusContextManager>,
    transaction_processor: Arc<PluginTransactionProcessor>,
}

impl PluginSystem {
    /// Create new plugin system with Q-VM integration
    pub async fn new(
        config: PluginSystemConfig,
        vm: Arc<q_vm::VirtualMachine>,
        consensus: Arc<q_vm::consensus::narwhal_bullshark::NarwhalBullshark>,
        state_access: Arc<dyn StateAccess>,
    ) -> Result<Self, PluginSystemError> {
        // Initialize core components
        let plugin_manager = Arc::new(PluginManager::new());
        let state_manager = Arc::new(PluginStateManager::new(state_access.clone()));

        // Create VM bridge
        let vm_bridge = Arc::new(PluginVmBridge::new(
            vm.clone(),
            consensus.clone(),
            plugin_manager.clone(),
            state_access.clone(),
            state_manager.clone(),
        ));

        // Create consensus context manager
        let consensus_context_manager = Arc::new(PluginConsensusContextManager::new(
            vm_bridge.clone(),
            consensus.clone(),
        ));

        // Create transaction processor
        let transaction_processor = Arc::new(PluginTransactionProcessor::new(
            vm_bridge.clone(),
            state_manager.clone(),
            consensus.clone(),
            vm.clone(),
        ));

        Ok(Self {
            config,
            plugin_manager,
            vm_bridge,
            state_manager,
            consensus_context_manager,
            transaction_processor,
        })
    }

    /// Register a new plugin in the system
    pub async fn register_plugin(
        &self,
        plugin_id: PluginId,
        plugin_config: PluginConfig,
        vm_permissions: VmPermissions,
        consensus_permissions: ConsensusPermissions,
        transaction_permissions: TransactionPermissions,
    ) -> Result<(), PluginSystemError> {
        // Register with plugin manager
        self.plugin_manager
            .register_plugin(plugin_id.clone(), plugin_config)
            .await?;

        // Register with VM bridge
        self.vm_bridge
            .register_plugin_for_vm(plugin_id.clone(), vm_permissions)
            .await?;

        // Register with consensus context manager
        if self.config.enable_consensus_integration {
            self.consensus_context_manager
                .register_plugin_for_consensus(
                    plugin_id.clone(),
                    consensus_permissions,
                    "default_validator".to_string(),
                )
                .await?;
        }

        // Register with transaction processor
        if self.config.enable_transaction_processing {
            self.transaction_processor
                .register_plugin_for_transactions(plugin_id.clone(), transaction_permissions)
                .await?;
        }

        Ok(())
    }

    /// Execute plugin in full system context
    pub async fn execute_plugin(
        &self,
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
    ) -> Result<Vec<u8>, PluginSystemError> {
        self.vm_bridge
            .execute_plugin_in_vm(plugin_id, hook, data, None)
            .await
            .map_err(|e| PluginSystemError::ExecutionFailed(e.to_string()))
    }

    /// Process transaction through plugin
    pub async fn process_transaction(
        &self,
        plugin_id: &PluginId,
        transaction_type: PluginTransactionType,
        sender: u64,
        recipient: Option<u64>,
        value: u64,
        gas_limit: u64,
        gas_price: u64,
        data: Vec<u8>,
    ) -> Result<ExecutionResult, PluginSystemError> {
        if !self.config.enable_transaction_processing {
            return Err(PluginSystemError::FeatureDisabled(
                "Transaction processing disabled".to_string(),
            ));
        }

        self.transaction_processor
            .process_plugin_transaction(
                plugin_id,
                transaction_type,
                sender,
                recipient,
                value,
                gas_limit,
                gas_price,
                data,
            )
            .await
            .map_err(|e| PluginSystemError::TransactionFailed(e.to_string()))
    }

    /// Get comprehensive plugin metrics
    pub async fn get_plugin_metrics(&self, plugin_id: &PluginId) -> Option<PluginMetrics> {
        let vm_metrics = self.vm_bridge.get_plugin_metrics(plugin_id).await?;
        let consensus_metrics = self
            .consensus_context_manager
            .get_consensus_metrics(plugin_id)
            .await;
        let transaction_metrics = self
            .transaction_processor
            .get_transaction_metrics(plugin_id)
            .await;

        Some(PluginMetrics {
            vm_metrics,
            consensus_metrics,
            transaction_metrics,
        })
    }

    /// Shutdown plugin system gracefully
    pub async fn shutdown(&self) -> Result<(), PluginSystemError> {
        // Implementation would gracefully shutdown all components
        Ok(())
    }
}

/// Combined plugin metrics from all subsystems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetrics {
    pub vm_metrics: PluginVmMetrics,
    pub consensus_metrics: Option<ConsensusOperationMetrics>,
    pub transaction_metrics: Option<TransactionProcessingMetrics>,
}

/// Plugin system errors
#[derive(Debug, thiserror::Error)]
pub enum PluginSystemError {
    #[error("Plugin execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Transaction processing failed: {0}")]
    TransactionFailed(String),

    #[error("Feature disabled: {0}")]
    FeatureDisabled(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Plugin error: {0}")]
    PluginError(#[from] PluginError),
}

/// Trait for consensus engine integration
pub trait ConsensusEngine: Send + Sync {
    // Define consensus interface methods that plugins can use
    fn get_current_height(&self) -> u64;
    fn validate_transaction(&self, tx_data: &[u8]) -> Result<bool, anyhow::Error>;
    fn propose_block(&self, block_data: Vec<u8>) -> Result<String, anyhow::Error>;
    fn get_network_state(&self) -> Result<String, anyhow::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_system_creation() {
        // Test plugin system initialization
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        // Test plugin registration flow
    }

    #[tokio::test]
    async fn test_plugin_execution() {
        // Test plugin execution through system
    }
}
