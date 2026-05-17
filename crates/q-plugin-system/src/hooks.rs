// Q-NarwhalKnight Plugin Hook System
//
// This module provides the event hook system that allows plugins to respond
// to blockchain events like blocks, transactions, and consensus rounds.
// It integrates with the executor and lifecycle modules for coordinated
// plugin execution.

use crate::lifecycle::{EntryPoint, PluginLifecycle, PluginStatus};
use crate::plugin::{PluginError, PluginHook, PluginId, PluginManager, PluginResult};
use crate::vm::PluginVmBridge;
use q_types::{QBlock, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ============================================================================
// HOOK EVENT TYPES
// ============================================================================

/// Event data passed to plugins for block events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockEvent {
    /// Block height
    pub height: u64,
    /// Block hash
    pub hash: [u8; 32],
    /// Block timestamp
    pub timestamp: u64,
    /// Number of transactions in block
    pub tx_count: usize,
    /// Block producer's public key
    pub producer: [u8; 32],
    /// Parent block hash
    pub parent_hash: [u8; 32],
    /// Raw block data (serialized)
    pub raw_data: Vec<u8>,
}

impl From<&QBlock> for BlockEvent {
    fn from(block: &QBlock) -> Self {
        Self {
            height: block.header.height,
            hash: block.hash,
            timestamp: block.header.timestamp,
            tx_count: block.transactions.len(),
            producer: block.header.producer,
            parent_hash: block.header.parent_hash,
            raw_data: bincode::serialize(block).unwrap_or_default(),
        }
    }
}

/// Event data passed to plugins for transaction events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionEvent {
    /// Transaction hash
    pub hash: [u8; 32],
    /// Sender address
    pub from: [u8; 32],
    /// Recipient address
    pub to: [u8; 32],
    /// Amount transferred
    pub amount: u128,
    /// Transaction fee
    pub fee: u128,
    /// Transaction nonce
    pub nonce: u64,
    /// Transaction type discriminant
    pub tx_type: u8,
    /// Raw transaction data
    pub raw_data: Vec<u8>,
}

impl From<&Transaction> for TransactionEvent {
    fn from(tx: &Transaction) -> Self {
        Self {
            hash: tx.hash,
            from: tx.from,
            to: tx.to,
            amount: tx.amount,
            fee: tx.fee,
            nonce: tx.nonce,
            tx_type: tx.tx_type.discriminant(),
            raw_data: bincode::serialize(tx).unwrap_or_default(),
        }
    }
}

/// Event data for consensus round events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRoundEvent {
    /// Round number
    pub round: u64,
    /// Current epoch
    pub epoch: u64,
    /// Current leader/anchor
    pub leader: Option<[u8; 32]>,
    /// Number of validators participating
    pub validator_count: usize,
    /// Round start timestamp
    pub started_at: u64,
    /// Additional consensus state data
    pub state_data: Vec<u8>,
}

/// Event data for peer events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerEvent {
    /// Peer ID (libp2p peer ID string)
    pub peer_id: String,
    /// Peer multiaddress
    pub multiaddr: String,
    /// Event type (connected/disconnected)
    pub event_type: PeerEventType,
    /// Timestamp
    pub timestamp: u64,
}

/// Peer event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeerEventType {
    Connected,
    Disconnected,
    IdentifyReceived,
}

/// Custom hook event for arbitrary plugin-defined events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHookEvent {
    /// Hook name
    pub name: String,
    /// Event payload
    pub data: Vec<u8>,
    /// Source plugin (if from another plugin)
    pub source: Option<PluginId>,
    /// Timestamp
    pub timestamp: u64,
}

// ============================================================================
// HOOK EXECUTION RESULT
// ============================================================================

/// Result of executing a hook across all plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookExecutionResult {
    /// Hook that was triggered
    pub entry_point: EntryPoint,
    /// Number of plugins invoked
    pub plugins_invoked: usize,
    /// Number of successful executions
    pub successful: usize,
    /// Number of failed executions
    pub failed: usize,
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
    /// Individual plugin results
    pub plugin_results: Vec<PluginHookResult>,
}

/// Result from a single plugin hook execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHookResult {
    /// Plugin ID
    pub plugin_id: PluginId,
    /// Whether execution succeeded
    pub success: bool,
    /// Return data from plugin
    pub return_data: Option<Vec<u8>>,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

// ============================================================================
// HOOK MANAGER
// ============================================================================

/// Hook manager that coordinates event triggering across plugins
pub struct HookManager {
    /// Plugin executor for running WASM
    executor: Arc<PluginVmBridge>,
    /// Lifecycle manager for plugin state
    lifecycle: Arc<RwLock<PluginLifecycle>>,
    /// Plugin manager for hook registration
    plugin_manager: Arc<PluginManager>,
    /// Hook execution statistics
    stats: Arc<RwLock<HookStats>>,
    /// Hook execution configuration
    config: HookManagerConfig,
}

/// Hook execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookStats {
    /// Total hooks triggered
    pub total_triggers: u64,
    /// Successful plugin invocations
    pub successful_invocations: u64,
    /// Failed plugin invocations
    pub failed_invocations: u64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Per-hook statistics
    pub per_hook_stats: HashMap<String, HookTypeStats>,
}

/// Statistics for a specific hook type
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookTypeStats {
    pub trigger_count: u64,
    pub avg_execution_time_ms: f64,
    pub last_triggered: Option<u64>,
}

/// Configuration for hook execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookManagerConfig {
    /// Maximum execution time per plugin (milliseconds)
    pub max_execution_time_ms: u64,
    /// Whether to continue execution if a plugin fails
    pub continue_on_error: bool,
    /// Maximum concurrent plugin executions
    pub max_concurrent_executions: usize,
    /// Whether to enable parallel execution
    pub parallel_execution: bool,
    /// Hook execution timeout for the entire batch
    pub batch_timeout_ms: u64,
}

impl Default for HookManagerConfig {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 5000,  // 5 seconds per plugin
            continue_on_error: true,
            max_concurrent_executions: 10,
            parallel_execution: false, // Sequential by default for determinism
            batch_timeout_ms: 30000,   // 30 seconds for all plugins
        }
    }
}

impl HookManager {
    /// Create a new hook manager
    pub fn new(
        executor: Arc<PluginVmBridge>,
        lifecycle: Arc<RwLock<PluginLifecycle>>,
        plugin_manager: Arc<PluginManager>,
        config: HookManagerConfig,
    ) -> Self {
        Self {
            executor,
            lifecycle,
            plugin_manager,
            stats: Arc::new(RwLock::new(HookStats::default())),
            config,
        }
    }

    /// Trigger OnBlockReceived hook for all registered plugins
    pub async fn trigger_block_received(&self, block: &QBlock) -> HookExecutionResult {
        let event = BlockEvent::from(block);
        let event_data = bincode::serialize(&event).unwrap_or_default();

        info!("Triggering OnBlockReceived hook for block {}", event.height);

        self.trigger_hook(EntryPoint::OnBlockReceived, &event_data).await
    }

    /// Trigger OnTransactionReceived hook for all registered plugins
    pub async fn trigger_transaction_received(&self, tx: &Transaction) -> HookExecutionResult {
        let event = TransactionEvent::from(tx);
        let event_data = bincode::serialize(&event).unwrap_or_default();

        debug!("Triggering OnTransactionReceived hook for tx {:?}", hex::encode(&event.hash[..8]));

        self.trigger_hook(EntryPoint::OnTransactionReceived, &event_data).await
    }

    /// Trigger OnConsensusRound hook for all registered plugins
    pub async fn trigger_consensus_round(&self, round: ConsensusRoundEvent) -> HookExecutionResult {
        let event_data = bincode::serialize(&round).unwrap_or_default();

        debug!("Triggering OnConsensusRound hook for round {}", round.round);

        self.trigger_hook(EntryPoint::OnConsensusRound, &event_data).await
    }

    /// Trigger BeforeTransactionExecution hook
    pub async fn trigger_before_tx_execution(&self, tx: &Transaction) -> HookExecutionResult {
        let event = TransactionEvent::from(tx);
        let event_data = bincode::serialize(&event).unwrap_or_default();

        self.trigger_hook(EntryPoint::BeforeTransactionExecution, &event_data).await
    }

    /// Trigger AfterTransactionExecution hook
    pub async fn trigger_after_tx_execution(
        &self,
        tx: &Transaction,
        success: bool,
    ) -> HookExecutionResult {
        #[derive(Serialize)]
        struct AfterTxEvent {
            tx: TransactionEvent,
            success: bool,
        }

        let event = AfterTxEvent {
            tx: TransactionEvent::from(tx),
            success,
        };
        let event_data = bincode::serialize(&event).unwrap_or_default();

        self.trigger_hook(EntryPoint::AfterTransactionExecution, &event_data).await
    }

    /// Trigger OnPeerConnected hook
    pub async fn trigger_peer_connected(&self, event: PeerEvent) -> HookExecutionResult {
        let event_data = bincode::serialize(&event).unwrap_or_default();
        self.trigger_hook(EntryPoint::OnPeerConnected, &event_data).await
    }

    /// Trigger OnPeerDisconnected hook
    pub async fn trigger_peer_disconnected(&self, event: PeerEvent) -> HookExecutionResult {
        let event_data = bincode::serialize(&event).unwrap_or_default();
        self.trigger_hook(EntryPoint::OnPeerDisconnected, &event_data).await
    }

    /// Trigger OnTimer hook for scheduled plugin executions
    pub async fn trigger_timer(&self) -> HookExecutionResult {
        let now = chrono::Utc::now().timestamp() as u64;
        let event_data = now.to_le_bytes().to_vec();
        self.trigger_hook(EntryPoint::OnTimer, &event_data).await
    }

    /// Trigger OnNodeStart hook
    pub async fn trigger_node_start(&self) -> HookExecutionResult {
        let event_data = Vec::new();
        self.trigger_hook(EntryPoint::OnNodeStart, &event_data).await
    }

    /// Trigger OnNodeShutdown hook
    pub async fn trigger_node_shutdown(&self) -> HookExecutionResult {
        let event_data = Vec::new();
        self.trigger_hook(EntryPoint::OnNodeShutdown, &event_data).await
    }

    /// Trigger a custom hook with arbitrary data
    pub async fn trigger_custom(&self, hook_name: &str, data: &[u8]) -> HookExecutionResult {
        let event = CustomHookEvent {
            name: hook_name.to_string(),
            data: data.to_vec(),
            source: None,
            timestamp: chrono::Utc::now().timestamp() as u64,
        };
        let event_data = bincode::serialize(&event).unwrap_or_default();

        self.trigger_hook(EntryPoint::Custom(hook_name.to_string()), &event_data).await
    }

    /// Trigger a hook for specific plugins only
    pub async fn trigger_for_plugins(
        &self,
        entry_point: EntryPoint,
        plugin_ids: &[PluginId],
        data: &[u8],
    ) -> HookExecutionResult {
        let start = Instant::now();
        let mut plugin_results = Vec::new();
        let mut successful = 0;
        let mut failed = 0;

        for plugin_id in plugin_ids {
            let result = self.execute_plugin_hook(plugin_id, &entry_point, data).await;

            if result.success {
                successful += 1;
            } else {
                failed += 1;
            }
            plugin_results.push(result);
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        // Update stats
        self.update_stats(&entry_point, successful, failed, total_time_ms).await;

        HookExecutionResult {
            entry_point,
            plugins_invoked: plugin_ids.len(),
            successful,
            failed,
            total_time_ms,
            plugin_results,
        }
    }

    /// Get hook execution statistics
    pub async fn get_stats(&self) -> HookStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = HookStats::default();
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    async fn trigger_hook(&self, entry_point: EntryPoint, data: &[u8]) -> HookExecutionResult {
        let start = Instant::now();

        // Get active plugins for this hook
        let lifecycle = self.lifecycle.read().await;
        let plugin_ids = lifecycle.get_active_plugins_for_hook(&entry_point).await;
        drop(lifecycle);

        if plugin_ids.is_empty() {
            return HookExecutionResult {
                entry_point,
                plugins_invoked: 0,
                successful: 0,
                failed: 0,
                total_time_ms: 0,
                plugin_results: vec![],
            };
        }

        let mut plugin_results = Vec::new();
        let mut successful = 0;
        let mut failed = 0;

        if self.config.parallel_execution {
            // Parallel execution using tokio
            let futures: Vec<_> = plugin_ids
                .iter()
                .map(|pid| {
                    let plugin_id = pid.clone();
                    let entry_point = entry_point.clone();
                    let data = data.to_vec();
                    let executor = self.executor.clone();
                    let plugin_manager = self.plugin_manager.clone();
                    let max_time = self.config.max_execution_time_ms;

                    async move {
                        let hook: PluginHook = entry_point.into();
                        Self::execute_single_plugin(
                            &plugin_id,
                            hook,
                            &data,
                            executor,
                            plugin_manager,
                            max_time,
                        ).await
                    }
                })
                .collect();

            let results = futures::future::join_all(futures).await;
            for result in results {
                if result.success {
                    successful += 1;
                } else {
                    failed += 1;
                }
                plugin_results.push(result);
            }
        } else {
            // Sequential execution
            for plugin_id in &plugin_ids {
                let result = self.execute_plugin_hook(plugin_id, &entry_point, data).await;

                if result.success {
                    successful += 1;
                } else {
                    failed += 1;
                    if !self.config.continue_on_error {
                        plugin_results.push(result);
                        break;
                    }
                }
                plugin_results.push(result);
            }
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        // Update stats
        self.update_stats(&entry_point, successful, failed, total_time_ms).await;

        HookExecutionResult {
            entry_point,
            plugins_invoked: plugin_ids.len(),
            successful,
            failed,
            total_time_ms,
            plugin_results,
        }
    }

    async fn execute_plugin_hook(
        &self,
        plugin_id: &PluginId,
        entry_point: &EntryPoint,
        data: &[u8],
    ) -> PluginHookResult {
        let hook: PluginHook = entry_point.clone().into();
        Self::execute_single_plugin(
            plugin_id,
            hook,
            data,
            self.executor.clone(),
            self.plugin_manager.clone(),
            self.config.max_execution_time_ms,
        ).await
    }

    async fn execute_single_plugin(
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
        executor: Arc<PluginVmBridge>,
        plugin_manager: Arc<PluginManager>,
        max_time_ms: u64,
    ) -> PluginHookResult {
        let start = Instant::now();

        // Execute with timeout
        let timeout = Duration::from_millis(max_time_ms);
        let execution = tokio::time::timeout(
            timeout,
            executor.execute_plugin_in_vm(plugin_id, hook.clone(), data, None),
        );

        match execution.await {
            Ok(Ok(return_data)) => {
                PluginHookResult {
                    plugin_id: plugin_id.clone(),
                    success: true,
                    return_data: Some(return_data),
                    error: None,
                    execution_time_ms: start.elapsed().as_millis() as u64,
                }
            }
            Ok(Err(e)) => {
                warn!("Plugin {} hook {:?} failed: {}", plugin_id, hook, e);
                PluginHookResult {
                    plugin_id: plugin_id.clone(),
                    success: false,
                    return_data: None,
                    error: Some(e.to_string()),
                    execution_time_ms: start.elapsed().as_millis() as u64,
                }
            }
            Err(_) => {
                error!("Plugin {} hook {:?} timed out", plugin_id, hook);
                PluginHookResult {
                    plugin_id: plugin_id.clone(),
                    success: false,
                    return_data: None,
                    error: Some(format!("Execution timed out after {}ms", max_time_ms)),
                    execution_time_ms: max_time_ms,
                }
            }
        }
    }

    async fn update_stats(
        &self,
        entry_point: &EntryPoint,
        successful: usize,
        failed: usize,
        total_time_ms: u64,
    ) {
        let mut stats = self.stats.write().await;
        stats.total_triggers += 1;
        stats.successful_invocations += successful as u64;
        stats.failed_invocations += failed as u64;
        stats.total_execution_time_ms += total_time_ms;

        let hook_key = format!("{:?}", entry_point);
        let hook_stats = stats.per_hook_stats.entry(hook_key).or_default();
        hook_stats.trigger_count += 1;
        hook_stats.last_triggered = Some(chrono::Utc::now().timestamp() as u64);

        // Update average execution time
        let total_invocations = successful + failed;
        if total_invocations > 0 {
            let current_avg = hook_stats.avg_execution_time_ms;
            let new_time = total_time_ms as f64 / total_invocations as f64;
            hook_stats.avg_execution_time_ms =
                (current_avg * (hook_stats.trigger_count - 1) as f64 + new_time)
                    / hook_stats.trigger_count as f64;
        }
    }
}

// ============================================================================
// HOOK FILTER
// ============================================================================

/// Filter for selecting which plugins receive a hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookFilter {
    /// Only include these plugins (empty = all)
    pub include_plugins: Vec<PluginId>,
    /// Exclude these plugins
    pub exclude_plugins: Vec<PluginId>,
    /// Only include plugins with these capabilities
    pub required_capabilities: Vec<String>,
}

impl Default for HookFilter {
    fn default() -> Self {
        Self {
            include_plugins: vec![],
            exclude_plugins: vec![],
            required_capabilities: vec![],
        }
    }
}

impl HookFilter {
    /// Create a filter that includes all plugins
    pub fn all() -> Self {
        Self::default()
    }

    /// Create a filter for specific plugins
    pub fn only(plugins: Vec<PluginId>) -> Self {
        Self {
            include_plugins: plugins,
            ..Default::default()
        }
    }

    /// Create a filter that excludes specific plugins
    pub fn except(plugins: Vec<PluginId>) -> Self {
        Self {
            exclude_plugins: plugins,
            ..Default::default()
        }
    }

    /// Check if a plugin passes this filter
    pub fn allows(&self, plugin_id: &PluginId) -> bool {
        // Check exclusion first
        if self.exclude_plugins.contains(plugin_id) {
            return false;
        }

        // Check inclusion
        if !self.include_plugins.is_empty() && !self.include_plugins.contains(plugin_id) {
            return false;
        }

        true
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_event_creation() {
        let event = BlockEvent {
            height: 1000,
            hash: [1u8; 32],
            timestamp: 12345,
            tx_count: 10,
            producer: [2u8; 32],
            parent_hash: [3u8; 32],
            raw_data: vec![],
        };

        let serialized = bincode::serialize(&event).unwrap();
        let deserialized: BlockEvent = bincode::deserialize(&serialized).unwrap();
        assert_eq!(event.height, deserialized.height);
        assert_eq!(event.tx_count, deserialized.tx_count);
    }

    #[test]
    fn test_hook_filter_allows() {
        let filter = HookFilter::only(vec!["plugin1".to_string(), "plugin2".to_string()]);
        assert!(filter.allows(&"plugin1".to_string()));
        assert!(filter.allows(&"plugin2".to_string()));
        assert!(!filter.allows(&"plugin3".to_string()));

        let filter = HookFilter::except(vec!["blocked".to_string()]);
        assert!(filter.allows(&"allowed".to_string()));
        assert!(!filter.allows(&"blocked".to_string()));

        let filter = HookFilter::all();
        assert!(filter.allows(&"any-plugin".to_string()));
    }

    #[test]
    fn test_hook_manager_config_defaults() {
        let config = HookManagerConfig::default();
        assert_eq!(config.max_execution_time_ms, 5000);
        assert!(config.continue_on_error);
        assert!(!config.parallel_execution);
    }

    #[test]
    fn test_hook_execution_result() {
        let result = HookExecutionResult {
            entry_point: EntryPoint::OnBlockReceived,
            plugins_invoked: 5,
            successful: 4,
            failed: 1,
            total_time_ms: 250,
            plugin_results: vec![],
        };

        assert_eq!(result.plugins_invoked, result.successful + result.failed);
    }

    #[test]
    fn test_consensus_round_event() {
        let event = ConsensusRoundEvent {
            round: 42,
            epoch: 5,
            leader: Some([1u8; 32]),
            validator_count: 21,
            started_at: chrono::Utc::now().timestamp() as u64,
            state_data: vec![],
        };

        let serialized = bincode::serialize(&event).unwrap();
        let deserialized: ConsensusRoundEvent = bincode::deserialize(&serialized).unwrap();
        assert_eq!(event.round, deserialized.round);
        assert_eq!(event.validator_count, deserialized.validator_count);
    }

    #[test]
    fn test_custom_hook_event() {
        let event = CustomHookEvent {
            name: "my-custom-hook".to_string(),
            data: vec![1, 2, 3, 4],
            source: Some("source-plugin".to_string()),
            timestamp: 12345,
        };

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: CustomHookEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(event.name, deserialized.name);
        assert_eq!(event.data, deserialized.data);
    }
}
