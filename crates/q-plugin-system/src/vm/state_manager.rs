use crate::plugin::{PluginError, PluginId, PluginResult, PluginState};
use crate::vm::{PluginVmContext, PluginVmMetrics};
use q_vm::vm::{ContractState, StateAccess, VmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Plugin State Manager for Q-NarwhalKnight VM Integration
///
/// This module provides comprehensive state management for plugins within the VM,
/// enabling persistent storage, state transitions, and consensus integration.
/// Adapted from Orobit Chimeras for Q-NarwhalKnight architecture.
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Plugin state management with VM integration
pub struct PluginStateManager {
    /// Active plugin states
    plugin_states: Arc<RwLock<HashMap<PluginId, Arc<Mutex<PluginState>>>>>,

    /// Plugin VM contexts
    vm_contexts: Arc<RwLock<HashMap<PluginId, PluginVmContext>>>,

    /// Plugin execution metrics
    metrics: Arc<RwLock<HashMap<PluginId, PluginVmMetrics>>>,

    /// Persistent state storage
    state_storage: Arc<dyn StateAccess>,

    /// State transition history
    state_history: Arc<RwLock<HashMap<PluginId, Vec<StateTransition>>>>,

    /// Global state locks for atomic operations
    state_locks: Arc<RwLock<HashMap<PluginId, Arc<Mutex<()>>>>>,
}

/// State transition record for audit and rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub timestamp: u64,
    pub transition_type: StateTransitionType,
    pub old_state_hash: String,
    pub new_state_hash: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateTransitionType {
    Initialization,
    Update,
    Consensus,
    Rollback,
    Migration,
}

/// Plugin state snapshot for persistence and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStateSnapshot {
    pub plugin_id: PluginId,
    pub state: PluginState,
    pub vm_context: Option<PluginVmContext>,
    pub metrics: PluginVmMetrics,
    pub timestamp: u64,
    pub state_hash: String,
}

impl PluginStateManager {
    pub fn new(state_storage: Arc<dyn StateAccess>) -> Self {
        Self {
            plugin_states: Arc::new(RwLock::new(HashMap::new())),
            vm_contexts: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            state_storage,
            state_history: Arc::new(RwLock::new(HashMap::new())),
            state_locks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize plugin state with VM context
    pub async fn initialize_plugin_state(
        &self,
        plugin_id: PluginId,
        initial_state: PluginState,
        vm_context: PluginVmContext,
    ) -> PluginResult<()> {
        // Acquire state lock
        let lock_arc = self.acquire_state_lock(&plugin_id).await;
        let _lock = lock_arc.lock().await;

        // Initialize state
        let state_arc = Arc::new(Mutex::new(initial_state.clone()));
        let metrics = PluginVmMetrics::default();

        // Store in memory
        {
            let mut states = self.plugin_states.write().await;
            states.insert(plugin_id.clone(), state_arc);
        }

        {
            let mut contexts = self.vm_contexts.write().await;
            contexts.insert(plugin_id.clone(), vm_context);
        }

        {
            let mut metrics_map = self.metrics.write().await;
            metrics_map.insert(plugin_id.clone(), metrics);
        }

        // Persist to storage
        self.persist_plugin_state(&plugin_id).await?;

        // Record state transition
        self.record_state_transition(
            &plugin_id,
            StateTransitionType::Initialization,
            "".to_string(),
            self.compute_state_hash(&initial_state).await?,
        )
        .await?;

        Ok(())
    }

    /// Get plugin state with automatic loading from storage
    pub async fn get_plugin_state(&self, plugin_id: &PluginId) -> Option<Arc<Mutex<PluginState>>> {
        // Try memory first
        {
            let states = self.plugin_states.read().await;
            if let Some(state) = states.get(plugin_id) {
                return Some(state.clone());
            }
        }

        // Try loading from storage
        if let Ok(Some(snapshot)) = self.load_plugin_state_from_storage(plugin_id).await {
            let state_arc = Arc::new(Mutex::new(snapshot.state));

            // Cache in memory
            {
                let mut states = self.plugin_states.write().await;
                states.insert(plugin_id.clone(), state_arc.clone());
            }

            if let Some(vm_context) = snapshot.vm_context {
                let mut contexts = self.vm_contexts.write().await;
                contexts.insert(plugin_id.clone(), vm_context);
            }

            {
                let mut metrics = self.metrics.write().await;
                metrics.insert(plugin_id.clone(), snapshot.metrics);
            }

            Some(state_arc)
        } else {
            None
        }
    }

    /// Update plugin state atomically
    pub async fn update_plugin_state<F, R>(
        &self,
        plugin_id: &PluginId,
        update_fn: F,
    ) -> PluginResult<R>
    where
        F: FnOnce(&mut PluginState) -> PluginResult<R> + Send,
    {
        // Acquire state lock
        let lock_arc = self.acquire_state_lock(plugin_id).await;
        let _lock = lock_arc.lock().await;

        let state_arc = self
            .get_plugin_state(plugin_id)
            .await
            .ok_or_else(|| PluginError::NotFound(format!("{:?}", plugin_id)))?;

        let old_state_hash = {
            let state = state_arc.lock().await;
            self.compute_state_hash(&*state).await?
        };

        // Apply update
        let result = {
            let mut state = state_arc.lock().await;
            update_fn(&mut *state)?
        };

        let new_state_hash = {
            let state = state_arc.lock().await;
            self.compute_state_hash(&*state).await?
        };

        // Persist changes
        self.persist_plugin_state(plugin_id).await?;

        // Record state transition
        self.record_state_transition(
            plugin_id,
            StateTransitionType::Update,
            old_state_hash,
            new_state_hash,
        )
        .await?;

        Ok(result)
    }

    /// Create state snapshot for backup/migration
    pub async fn create_state_snapshot(
        &self,
        plugin_id: &PluginId,
    ) -> PluginResult<PluginStateSnapshot> {
        let state_arc = self
            .get_plugin_state(plugin_id)
            .await
            .ok_or_else(|| PluginError::NotFound(format!("{:?}", plugin_id)))?;

        let state = {
            let state_guard = state_arc.lock().await;
            state_guard.clone()
        };

        let vm_context = self.get_vm_context(plugin_id).await;
        let metrics = self.get_metrics(plugin_id).await.unwrap_or_default();
        let state_hash = self.compute_state_hash(&state).await?;

        Ok(PluginStateSnapshot {
            plugin_id: plugin_id.clone(),
            state,
            vm_context,
            metrics,
            timestamp: chrono::Utc::now().timestamp() as u64,
            state_hash,
        })
    }

    /// Remove plugin state
    pub async fn remove_plugin_state(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let lock_arc = self.acquire_state_lock(plugin_id).await;
        let _lock = lock_arc.lock().await;

        // Remove from memory
        {
            let mut states = self.plugin_states.write().await;
            states.remove(plugin_id);
        }

        {
            let mut contexts = self.vm_contexts.write().await;
            contexts.remove(plugin_id);
        }

        {
            let mut metrics = self.metrics.write().await;
            metrics.remove(plugin_id);
        }

        // Remove from persistent storage
        self.remove_plugin_state_from_storage(plugin_id).await?;

        // Clean up state history
        {
            let mut history = self.state_history.write().await;
            history.remove(plugin_id);
        }

        // Remove state lock
        {
            let mut locks = self.state_locks.write().await;
            locks.remove(plugin_id);
        }

        Ok(())
    }

    /// Get state transition history
    pub async fn get_state_history(&self, plugin_id: &PluginId) -> Vec<StateTransition> {
        let history = self.state_history.read().await;
        history.get(plugin_id).cloned().unwrap_or_default()
    }

    /// List all managed plugin states
    pub async fn list_plugin_states(&self) -> Vec<PluginId> {
        let states = self.plugin_states.read().await;
        states.keys().cloned().collect()
    }

    /// Get plugin VM context
    pub async fn get_vm_context(&self, plugin_id: &PluginId) -> Option<PluginVmContext> {
        let contexts = self.vm_contexts.read().await;
        contexts.get(plugin_id).cloned()
    }

    /// Get plugin metrics
    pub async fn get_metrics(&self, plugin_id: &PluginId) -> Option<PluginVmMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(plugin_id).cloned()
    }

    // Private helper methods

    async fn acquire_state_lock(&self, plugin_id: &PluginId) -> Arc<Mutex<()>> {
        // Get or create lock for this plugin
        let lock_arc = {
            let mut locks = self.state_locks.write().await;
            locks
                .entry(plugin_id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        lock_arc
    }

    async fn persist_plugin_state(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let snapshot = self.create_state_snapshot(plugin_id).await?;
        let serialized = bincode::serialize(&snapshot)
            .map_err(|e| PluginError::Internal(format!("Serialization failed: {}", e)))?;

        // Store in VM state using plugin ID as key
        let storage_key = format!("plugin_state_{}", plugin_id).into_bytes();
        self.state_storage
            .set_storage(0, storage_key, serialized)
            .await
            .map_err(|e| PluginError::Internal(format!("Storage failed: {}", e)))?;

        Ok(())
    }

    async fn load_plugin_state_from_storage(
        &self,
        plugin_id: &PluginId,
    ) -> PluginResult<Option<PluginStateSnapshot>> {
        let storage_key = format!("plugin_state_{}", plugin_id).into_bytes();

        match self.state_storage.get_storage(0, &storage_key).await {
            Ok(Some(data)) => {
                let snapshot: PluginStateSnapshot = bincode::deserialize(&data)
                    .map_err(|e| PluginError::Internal(format!("Deserialization failed: {}", e)))?;
                Ok(Some(snapshot))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PluginError::Internal(format!("Storage read failed: {}", e))),
        }
    }

    async fn remove_plugin_state_from_storage(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let storage_key = format!("plugin_state_{}", plugin_id).into_bytes();

        // Set empty value to effectively remove
        self.state_storage
            .set_storage(0, storage_key, Vec::new())
            .await
            .map_err(|e| PluginError::Internal(format!("Storage removal failed: {}", e)))?;

        Ok(())
    }

    async fn compute_state_hash(&self, state: &PluginState) -> PluginResult<String> {
        let serialized = bincode::serialize(state)
            .map_err(|e| PluginError::Internal(format!("State serialization failed: {}", e)))?;

        let hash = blake3::hash(&serialized);
        Ok(hex::encode(hash.as_bytes()))
    }

    async fn record_state_transition(
        &self,
        plugin_id: &PluginId,
        transition_type: StateTransitionType,
        old_state_hash: String,
        new_state_hash: String,
    ) -> PluginResult<()> {
        let transition = StateTransition {
            timestamp: chrono::Utc::now().timestamp() as u64,
            transition_type,
            old_state_hash,
            new_state_hash,
            metadata: HashMap::new(),
        };

        let mut history = self.state_history.write().await;
        let plugin_history = history.entry(plugin_id.clone()).or_insert_with(Vec::new);
        plugin_history.push(transition);

        // Keep only the last 100 transitions to prevent unbounded growth
        if plugin_history.len() > 100 {
            plugin_history.drain(0..plugin_history.len() - 100);
        }

        Ok(())
    }
}
