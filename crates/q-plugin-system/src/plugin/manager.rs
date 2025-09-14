use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Plugin Manager for Q-NarwhalKnight Plugin System
///
/// Core plugin management functionality adapted from Orobit Chimeras
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::plugin::{
    NetworkStats, PluginConfig, PluginError, PluginExecutionContext, PluginHook, PluginId,
    PluginPermissions, PluginResourceLimits, PluginResult, PluginState, ResourceUsage,
};

/// Plugin manager for lifecycle and execution
pub struct PluginManager {
    /// Registered plugins
    plugins: Arc<RwLock<HashMap<PluginId, PluginInfo>>>,

    /// Plugin execution states
    plugin_states: Arc<RwLock<HashMap<PluginId, Arc<Mutex<PluginState>>>>>,

    /// Plugin execution locks
    execution_locks: Arc<RwLock<HashMap<PluginId, Arc<Mutex<()>>>>>,

    /// Plugin hooks registry
    hooks: Arc<RwLock<HashMap<PluginHook, Vec<PluginId>>>>,
}

/// Plugin information stored by manager
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub config: PluginConfig,
    pub status: PluginStatus,
    pub load_time: u64,
    pub last_execution: Option<u64>,
    pub execution_count: u64,
    pub error_count: u64,
}

/// Plugin status tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginStatus {
    Loaded,
    Running,
    Stopped,
    Error(String),
    Unloaded,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            plugin_states: Arc::new(RwLock::new(HashMap::new())),
            execution_locks: Arc::new(RwLock::new(HashMap::new())),
            hooks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new plugin
    pub async fn register_plugin(
        &self,
        plugin_id: PluginId,
        config: PluginConfig,
    ) -> PluginResult<()> {
        let mut plugins = self.plugins.write().await;

        if plugins.contains_key(&plugin_id) {
            return Err(PluginError::AlreadyExists(plugin_id));
        }

        // Create plugin info
        let info = PluginInfo {
            config: config.clone(),
            status: PluginStatus::Loaded,
            load_time: chrono::Utc::now().timestamp() as u64,
            last_execution: None,
            execution_count: 0,
            error_count: 0,
        };

        // Register plugin
        plugins.insert(plugin_id.clone(), info);

        // Initialize plugin state
        let initial_state = PluginState {
            storage: HashMap::new(),
            resource_usage: ResourceUsage::default(),
            network_stats: NetworkStats::default(),
        };

        {
            let mut states = self.plugin_states.write().await;
            states.insert(plugin_id.clone(), Arc::new(Mutex::new(initial_state)));
        }

        // Create execution lock
        {
            let mut locks = self.execution_locks.write().await;
            locks.insert(plugin_id.clone(), Arc::new(Mutex::new(())));
        }

        // Execute load hook
        self.execute_hook(&plugin_id, PluginHook::OnLoad, &[])
            .await
            .ok();

        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        // Execute unload hook first
        self.execute_hook(plugin_id, PluginHook::OnUnload, &[])
            .await
            .ok();

        // Remove from all data structures
        {
            let mut plugins = self.plugins.write().await;
            plugins.remove(plugin_id);
        }

        {
            let mut states = self.plugin_states.write().await;
            states.remove(plugin_id);
        }

        {
            let mut locks = self.execution_locks.write().await;
            locks.remove(plugin_id);
        }

        // Remove from hooks
        {
            let mut hooks = self.hooks.write().await;
            for (_, plugin_list) in hooks.iter_mut() {
                plugin_list.retain(|id| id != plugin_id);
            }
        }

        Ok(())
    }

    /// Execute a plugin hook
    pub async fn execute_hook(
        &self,
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
    ) -> PluginResult<Vec<u8>> {
        // Acquire execution lock
        let lock_arc = self.acquire_execution_lock(plugin_id).await?;
        let _lock = lock_arc.lock().await;

        // Get plugin info
        let (config, resource_limits) = {
            let plugins = self.plugins.read().await;
            let info = plugins
                .get(plugin_id)
                .ok_or_else(|| PluginError::not_found(plugin_id))?;
            (info.config.clone(), info.config.resource_limits.clone())
        };

        // Create execution context
        let context = PluginExecutionContext {
            plugin_id: plugin_id.clone(),
            execution_timestamp: chrono::Utc::now().timestamp() as u64,
            available_resources: resource_limits,
            permissions: config.permissions.clone(),
        };

        // Execute plugin with resource monitoring
        let result = self
            .execute_plugin_with_monitoring(plugin_id, hook, data, context)
            .await;

        // Update plugin statistics
        self.update_plugin_stats(plugin_id, &result).await;

        result
    }

    /// Execute plugin with enhanced VM context
    pub async fn execute_hook_with_context<T>(
        &self,
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
        _context: T, // Generic context for VM bridge integration
    ) -> PluginResult<Vec<u8>> {
        // For now, delegate to regular hook execution
        // This method provides extension point for VM bridge integration
        self.execute_hook(plugin_id, hook, data).await
    }

    /// Register plugin for specific hook
    pub async fn register_hook(&self, plugin_id: PluginId, hook: PluginHook) -> PluginResult<()> {
        let mut hooks = self.hooks.write().await;
        let hook_plugins = hooks.entry(hook).or_insert_with(Vec::new);

        if !hook_plugins.contains(&plugin_id) {
            hook_plugins.push(plugin_id);
        }

        Ok(())
    }

    /// Execute all plugins registered for a hook
    pub async fn execute_all_hooks(
        &self,
        hook: PluginHook,
        data: &[u8],
    ) -> Vec<(PluginId, PluginResult<Vec<u8>>)> {
        let plugin_ids = {
            let hooks = self.hooks.read().await;
            hooks.get(&hook).cloned().unwrap_or_default()
        };

        let mut results = Vec::new();
        for plugin_id in plugin_ids {
            let result = self.execute_hook(&plugin_id, hook.clone(), data).await;
            results.push((plugin_id, result));
        }

        results
    }

    /// Get plugin information
    pub async fn get_plugin_info(&self, plugin_id: &PluginId) -> Option<PluginInfo> {
        let plugins = self.plugins.read().await;
        plugins.get(plugin_id).cloned()
    }

    /// Get plugin state
    pub async fn get_plugin_state(&self, plugin_id: &PluginId) -> Option<Arc<Mutex<PluginState>>> {
        let states = self.plugin_states.read().await;
        states.get(plugin_id).cloned()
    }

    /// List all registered plugins
    pub async fn list_plugins(&self) -> Vec<(PluginId, PluginInfo)> {
        let plugins = self.plugins.read().await;
        plugins
            .iter()
            .map(|(id, info)| (id.clone(), info.clone()))
            .collect()
    }

    /// Update plugin state
    pub async fn update_plugin_state<F, R>(
        &self,
        plugin_id: &PluginId,
        update_fn: F,
    ) -> PluginResult<R>
    where
        F: FnOnce(&mut PluginState) -> R + Send,
    {
        let state_arc = self
            .get_plugin_state(plugin_id)
            .await
            .ok_or_else(|| PluginError::not_found(plugin_id))?;

        let mut state = state_arc.lock().await;
        Ok(update_fn(&mut *state))
    }

    /// Get plugin metrics
    pub async fn get_plugin_metrics(&self, plugin_id: &PluginId) -> Option<PluginMetrics> {
        let info = self.get_plugin_info(plugin_id).await?;
        let state_arc = self.get_plugin_state(plugin_id).await?;
        let state = state_arc.lock().await;

        Some(PluginMetrics {
            execution_count: info.execution_count,
            error_count: info.error_count,
            last_execution: info.last_execution,
            resource_usage: state.resource_usage.clone(),
            network_stats: state.network_stats.clone(),
        })
    }

    // Private helper methods

    async fn acquire_execution_lock(&self, plugin_id: &PluginId) -> PluginResult<Arc<Mutex<()>>> {
        let locks = self.execution_locks.read().await;
        locks
            .get(plugin_id)
            .cloned()
            .ok_or_else(|| PluginError::not_found(plugin_id))
    }

    async fn execute_plugin_with_monitoring(
        &self,
        plugin_id: &PluginId,
        hook: PluginHook,
        data: &[u8],
        _context: PluginExecutionContext,
    ) -> PluginResult<Vec<u8>> {
        // For this implementation, we'll simulate plugin execution
        // In a real implementation, this would:
        // 1. Load and execute actual plugin code (WASM, native, etc.)
        // 2. Monitor resource usage
        // 3. Enforce timeouts and limits
        // 4. Handle security sandboxing

        let start_time = std::time::Instant::now();

        // Simulate execution based on hook type
        let result = match hook {
            PluginHook::OnLoad => Ok(format!("Plugin {} loaded", plugin_id).into_bytes()),
            PluginHook::OnUnload => Ok(format!("Plugin {} unloaded", plugin_id).into_bytes()),
            PluginHook::BeforeTransaction => {
                Ok(format!("Plugin {} pre-transaction processing", plugin_id).into_bytes())
            }
            PluginHook::AfterTransaction => {
                Ok(format!("Plugin {} post-transaction processing", plugin_id).into_bytes())
            }
            PluginHook::BeforeBlock => {
                Ok(format!("Plugin {} pre-block processing", plugin_id).into_bytes())
            }
            PluginHook::AfterBlock => {
                Ok(format!("Plugin {} post-block processing", plugin_id).into_bytes())
            }
            PluginHook::OnConsensusEvent => {
                Ok(format!("Plugin {} consensus event handling", plugin_id).into_bytes())
            }
            PluginHook::OnNetworkEvent => {
                Ok(format!("Plugin {} network event handling", plugin_id).into_bytes())
            }
            PluginHook::OnStateChange => {
                Ok(format!("Plugin {} state change handling", plugin_id).into_bytes())
            }
            PluginHook::Custom(name) => {
                Ok(format!("Plugin {} custom hook: {}", plugin_id, name).into_bytes())
            }
        };

        // Update resource usage
        let execution_time = start_time.elapsed();
        self.update_resource_usage(plugin_id, execution_time, data.len())
            .await;

        result
    }

    async fn update_resource_usage(
        &self,
        plugin_id: &PluginId,
        execution_time: std::time::Duration,
        data_size: usize,
    ) {
        if let Some(state_arc) = self.get_plugin_state(plugin_id).await {
            let mut state = state_arc.lock().await;
            state.resource_usage.execution_time_ms += execution_time.as_millis() as u64;
            state.resource_usage.memory_used += data_size as u64; // Simplified memory tracking
        }
    }

    async fn update_plugin_stats(&self, plugin_id: &PluginId, result: &PluginResult<Vec<u8>>) {
        let mut plugins = self.plugins.write().await;
        if let Some(info) = plugins.get_mut(plugin_id) {
            info.execution_count += 1;
            info.last_execution = Some(chrono::Utc::now().timestamp() as u64);

            match result {
                Ok(_) => {
                    info.status = PluginStatus::Running;
                }
                Err(e) => {
                    info.error_count += 1;
                    info.status = PluginStatus::Error(e.to_string());
                }
            }
        }
    }
}

/// Plugin execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetrics {
    pub execution_count: u64,
    pub error_count: u64,
    pub last_execution: Option<u64>,
    pub resource_usage: ResourceUsage,
    pub network_stats: NetworkStats,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}
