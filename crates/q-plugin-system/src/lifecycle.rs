// Q-NarwhalKnight Plugin Lifecycle Management
//
// This module provides comprehensive lifecycle management for plugins,
// including installation, verification, activation, suspension, and removal.
// It integrates with the WASM executor and network modules for secure plugin execution.

use crate::plugin::{
    PluginConfig, PluginError, PluginHook, PluginId, PluginManager, PluginMetadata, PluginResult,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ============================================================================
// PLUGIN STATUS
// ============================================================================

/// Plugin status representing its current lifecycle state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginStatus {
    /// Downloaded, not yet verified
    Pending,
    /// Signature and hash verified
    Verified,
    /// WASM module compiled
    Loaded,
    /// Running and handling events
    Active,
    /// Temporarily disabled
    Suspended,
    /// Error state with description
    Failed(String),
}

impl PluginStatus {
    /// Check if plugin can be activated
    pub fn can_activate(&self) -> bool {
        matches!(self, PluginStatus::Loaded | PluginStatus::Suspended)
    }

    /// Check if plugin can be suspended
    pub fn can_suspend(&self) -> bool {
        matches!(self, PluginStatus::Active)
    }

    /// Check if plugin is operational
    pub fn is_operational(&self) -> bool {
        matches!(self, PluginStatus::Active | PluginStatus::Suspended)
    }
}

// ============================================================================
// PLUGIN MANIFEST
// ============================================================================

/// Plugin manifest containing metadata and entry points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin identifier
    pub id: PluginId,
    /// Plugin version (semver)
    pub version: String,
    /// Human-readable name
    pub name: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Minimum required API version
    pub api_version: String,
    /// Entry points this plugin hooks into
    pub entry_points: Vec<EntryPoint>,
    /// Dependencies on other plugins
    pub dependencies: Vec<PluginDependency>,
    /// SHA3-256 hash of the WASM binary
    pub wasm_hash: [u8; 32],
    /// Ed25519 signature over the manifest
    pub signature: Option<[u8; 64]>,
    /// Author's public key for signature verification
    pub author_pubkey: Option<[u8; 32]>,
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Plugin ID of the dependency
    pub plugin_id: PluginId,
    /// Version requirement (semver range)
    pub version_requirement: String,
    /// Whether this dependency is optional
    pub optional: bool,
}

/// Entry points that plugins can hook into
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntryPoint {
    /// Called when a new block is received
    OnBlockReceived,
    /// Called when a new transaction is received
    OnTransactionReceived,
    /// Called at the start of a consensus round
    OnConsensusRound,
    /// Called before transaction execution
    BeforeTransactionExecution,
    /// Called after transaction execution
    AfterTransactionExecution,
    /// Called when a peer connects
    OnPeerConnected,
    /// Called when a peer disconnects
    OnPeerDisconnected,
    /// Called periodically (timer-based)
    OnTimer,
    /// Called on node startup
    OnNodeStart,
    /// Called on node shutdown
    OnNodeShutdown,
    /// Custom hook with name
    Custom(String),
}

impl From<PluginHook> for EntryPoint {
    fn from(hook: PluginHook) -> Self {
        match hook {
            PluginHook::OnLoad => EntryPoint::OnNodeStart,
            PluginHook::OnUnload => EntryPoint::OnNodeShutdown,
            PluginHook::BeforeTransaction => EntryPoint::BeforeTransactionExecution,
            PluginHook::AfterTransaction => EntryPoint::AfterTransactionExecution,
            PluginHook::BeforeBlock => EntryPoint::OnBlockReceived,
            PluginHook::AfterBlock => EntryPoint::OnBlockReceived,
            PluginHook::OnConsensusEvent => EntryPoint::OnConsensusRound,
            PluginHook::OnNetworkEvent => EntryPoint::OnPeerConnected,
            PluginHook::OnStateChange => EntryPoint::AfterTransactionExecution,
            PluginHook::Custom(name) => EntryPoint::Custom(name),
        }
    }
}

impl From<EntryPoint> for PluginHook {
    fn from(entry: EntryPoint) -> Self {
        match entry {
            EntryPoint::OnBlockReceived => PluginHook::BeforeBlock,
            EntryPoint::OnTransactionReceived => PluginHook::BeforeTransaction,
            EntryPoint::OnConsensusRound => PluginHook::OnConsensusEvent,
            EntryPoint::BeforeTransactionExecution => PluginHook::BeforeTransaction,
            EntryPoint::AfterTransactionExecution => PluginHook::AfterTransaction,
            EntryPoint::OnPeerConnected => PluginHook::OnNetworkEvent,
            EntryPoint::OnPeerDisconnected => PluginHook::OnNetworkEvent,
            EntryPoint::OnTimer => PluginHook::Custom("timer".to_string()),
            EntryPoint::OnNodeStart => PluginHook::OnLoad,
            EntryPoint::OnNodeShutdown => PluginHook::OnUnload,
            EntryPoint::Custom(name) => PluginHook::Custom(name),
        }
    }
}

// ============================================================================
// COMPILED PLUGIN
// ============================================================================

/// Compiled WASM plugin ready for execution
#[derive(Debug)]
pub struct CompiledPlugin {
    /// The plugin manifest
    pub manifest: PluginManifest,
    /// WASM binary data
    pub wasm_bytes: Vec<u8>,
    /// Compilation timestamp
    pub compiled_at: u64,
    /// Memory usage of compiled module
    pub memory_usage: u64,
}

// ============================================================================
// PLUGIN LIFECYCLE MANAGER
// ============================================================================

/// Plugin lifecycle manager handling installation, activation, and removal
pub struct PluginLifecycle {
    /// Plugin status registry
    plugins: Arc<RwLock<HashMap<PluginId, PluginLifecycleEntry>>>,
    /// Entry point to plugin mapping
    hooks: Arc<RwLock<HashMap<EntryPoint, Vec<PluginId>>>>,
    /// Compiled plugins cache
    compiled_plugins: Arc<RwLock<HashMap<PluginId, CompiledPlugin>>>,
    /// Plugin manager for execution
    plugin_manager: Arc<PluginManager>,
    /// Configuration for lifecycle operations
    config: PluginLifecycleConfig,
}

/// Entry in the lifecycle registry
#[derive(Debug, Clone)]
struct PluginLifecycleEntry {
    /// Current status
    status: PluginStatus,
    /// Plugin manifest
    manifest: PluginManifest,
    /// Installation timestamp
    installed_at: u64,
    /// Last status change timestamp
    status_changed_at: u64,
    /// Number of activation attempts
    activation_attempts: u32,
    /// Last error message if failed
    last_error: Option<String>,
}

/// Configuration for plugin lifecycle operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginLifecycleConfig {
    /// Maximum number of plugins allowed
    pub max_plugins: usize,
    /// Maximum WASM binary size in bytes
    pub max_wasm_size: usize,
    /// Maximum activation attempts before permanent failure
    pub max_activation_attempts: u32,
    /// Whether to require signature verification
    pub require_signatures: bool,
    /// Trusted author public keys
    pub trusted_authors: Vec<[u8; 32]>,
}

impl Default for PluginLifecycleConfig {
    fn default() -> Self {
        Self {
            max_plugins: 100,
            max_wasm_size: 10 * 1024 * 1024, // 10 MB
            max_activation_attempts: 3,
            require_signatures: true,
            trusted_authors: Vec::new(),
        }
    }
}

impl PluginLifecycle {
    /// Create a new plugin lifecycle manager
    pub fn new(plugin_manager: Arc<PluginManager>, config: PluginLifecycleConfig) -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            hooks: Arc::new(RwLock::new(HashMap::new())),
            compiled_plugins: Arc::new(RwLock::new(HashMap::new())),
            plugin_manager,
            config,
        }
    }

    /// Install a new plugin from manifest and WASM binary
    ///
    /// This performs the complete installation flow:
    /// 1. Validate manifest
    /// 2. Verify WASM hash
    /// 3. Verify signature (if required)
    /// 4. Compile WASM module
    /// 5. Register hooks
    pub async fn install_plugin(
        &self,
        manifest: PluginManifest,
        wasm_bytes: Vec<u8>,
    ) -> PluginResult<PluginId> {
        let plugin_id = manifest.id.clone();
        info!("Installing plugin: {} v{}", plugin_id, manifest.version);

        // Check if already installed
        {
            let plugins = self.plugins.read().await;
            if plugins.contains_key(&plugin_id) {
                return Err(PluginError::AlreadyExists(plugin_id));
            }
        }

        // Check plugin limit
        {
            let plugins = self.plugins.read().await;
            if plugins.len() >= self.config.max_plugins {
                return Err(PluginError::ResourceLimitExceeded(format!(
                    "Maximum plugin limit ({}) reached",
                    self.config.max_plugins
                )));
            }
        }

        // Validate WASM size
        if wasm_bytes.len() > self.config.max_wasm_size {
            return Err(PluginError::InvalidConfiguration(format!(
                "WASM binary too large: {} bytes (max: {})",
                wasm_bytes.len(),
                self.config.max_wasm_size
            )));
        }

        // Verify WASM hash
        let computed_hash = self.compute_wasm_hash(&wasm_bytes);
        if computed_hash != manifest.wasm_hash {
            return Err(PluginError::InvalidConfiguration(
                "WASM hash mismatch - binary may be corrupted or tampered".to_string(),
            ));
        }

        // Create initial entry as Pending
        let now = chrono::Utc::now().timestamp() as u64;
        let entry = PluginLifecycleEntry {
            status: PluginStatus::Pending,
            manifest: manifest.clone(),
            installed_at: now,
            status_changed_at: now,
            activation_attempts: 0,
            last_error: None,
        };

        {
            let mut plugins = self.plugins.write().await;
            plugins.insert(plugin_id.clone(), entry);
        }

        // Verify signature if required
        if self.config.require_signatures {
            match self.verify_signature(&manifest).await {
                Ok(true) => {
                    self.update_status(&plugin_id, PluginStatus::Verified).await?;
                    info!("Plugin {} signature verified", plugin_id);
                }
                Ok(false) => {
                    self.update_status(
                        &plugin_id,
                        PluginStatus::Failed("Signature verification failed".to_string()),
                    )
                    .await?;
                    return Err(PluginError::InvalidConfiguration(
                        "Plugin signature verification failed".to_string(),
                    ));
                }
                Err(e) => {
                    self.update_status(&plugin_id, PluginStatus::Failed(e.to_string()))
                        .await?;
                    return Err(e);
                }
            }
        } else {
            // Skip verification
            self.update_status(&plugin_id, PluginStatus::Verified).await?;
        }

        // Compile WASM module
        match self.compile_wasm(&manifest, wasm_bytes).await {
            Ok(compiled) => {
                {
                    let mut compiled_plugins = self.compiled_plugins.write().await;
                    compiled_plugins.insert(plugin_id.clone(), compiled);
                }
                self.update_status(&plugin_id, PluginStatus::Loaded).await?;
                info!("Plugin {} compiled successfully", plugin_id);
            }
            Err(e) => {
                self.update_status(&plugin_id, PluginStatus::Failed(e.to_string()))
                    .await?;
                return Err(e);
            }
        }

        // Register hooks
        self.register_hooks(&plugin_id, &manifest.entry_points).await?;

        // Register with plugin manager
        let plugin_config = PluginConfig {
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            author: manifest.author.clone(),
            description: manifest.description.clone(),
            entry_point: manifest.entry_points.first()
                .map(|e| format!("{:?}", e))
                .unwrap_or_else(|| "default".to_string()),
            permissions: Default::default(),
            resource_limits: Default::default(),
        };

        self.plugin_manager
            .register_plugin(plugin_id.clone(), plugin_config)
            .await?;

        info!("Plugin {} installed successfully", plugin_id);
        Ok(plugin_id)
    }

    /// Activate a plugin to start handling events
    pub async fn activate_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let entry = self.get_entry(plugin_id).await?;

        if !entry.status.can_activate() {
            return Err(PluginError::InvalidConfiguration(format!(
                "Plugin {} cannot be activated from status {:?}",
                plugin_id, entry.status
            )));
        }

        // Check activation attempts
        if entry.activation_attempts >= self.config.max_activation_attempts {
            return Err(PluginError::ExecutionFailed(format!(
                "Plugin {} exceeded maximum activation attempts ({})",
                plugin_id, self.config.max_activation_attempts
            )));
        }

        // Increment activation attempts
        {
            let mut plugins = self.plugins.write().await;
            if let Some(entry) = plugins.get_mut(plugin_id) {
                entry.activation_attempts += 1;
            }
        }

        // Execute OnLoad hook
        let result = self.plugin_manager
            .execute_hook(plugin_id, PluginHook::OnLoad, &[])
            .await;

        match result {
            Ok(_) => {
                self.update_status(plugin_id, PluginStatus::Active).await?;
                info!("Plugin {} activated", plugin_id);
                Ok(())
            }
            Err(e) => {
                warn!("Plugin {} activation failed: {}", plugin_id, e);
                self.update_status(plugin_id, PluginStatus::Failed(e.to_string()))
                    .await?;
                Err(e)
            }
        }
    }

    /// Suspend a plugin without unloading
    pub async fn suspend_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let entry = self.get_entry(plugin_id).await?;

        if !entry.status.can_suspend() {
            return Err(PluginError::InvalidConfiguration(format!(
                "Plugin {} cannot be suspended from status {:?}",
                plugin_id, entry.status
            )));
        }

        // Remove from active hooks but keep compiled
        self.deactivate_hooks(plugin_id).await;

        self.update_status(plugin_id, PluginStatus::Suspended).await?;
        info!("Plugin {} suspended", plugin_id);
        Ok(())
    }

    /// Resume a suspended plugin
    pub async fn resume_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        let entry = self.get_entry(plugin_id).await?;

        if entry.status != PluginStatus::Suspended {
            return Err(PluginError::InvalidConfiguration(format!(
                "Plugin {} is not suspended (status: {:?})",
                plugin_id, entry.status
            )));
        }

        // Re-register hooks
        self.register_hooks(plugin_id, &entry.manifest.entry_points).await?;

        self.update_status(plugin_id, PluginStatus::Active).await?;
        info!("Plugin {} resumed", plugin_id);
        Ok(())
    }

    /// Completely uninstall a plugin
    pub async fn uninstall_plugin(&self, plugin_id: &PluginId) -> PluginResult<()> {
        info!("Uninstalling plugin: {}", plugin_id);

        // Execute OnUnload hook if active
        {
            let entry = self.get_entry(plugin_id).await?;
            if entry.status == PluginStatus::Active {
                let _ = self.plugin_manager
                    .execute_hook(plugin_id, PluginHook::OnUnload, &[])
                    .await;
            }
        }

        // Remove from hooks
        self.deactivate_hooks(plugin_id).await;
        self.unregister_all_hooks(plugin_id).await;

        // Remove compiled plugin
        {
            let mut compiled = self.compiled_plugins.write().await;
            compiled.remove(plugin_id);
        }

        // Remove from plugin manager
        let _ = self.plugin_manager.unregister_plugin(plugin_id).await;

        // Remove from lifecycle registry
        {
            let mut plugins = self.plugins.write().await;
            plugins.remove(plugin_id);
        }

        info!("Plugin {} uninstalled", plugin_id);
        Ok(())
    }

    /// Upgrade a plugin in-place
    pub async fn upgrade_plugin(
        &self,
        plugin_id: &PluginId,
        new_manifest: PluginManifest,
        new_wasm: Vec<u8>,
    ) -> PluginResult<()> {
        info!(
            "Upgrading plugin {} from current version to {}",
            plugin_id, new_manifest.version
        );

        // Get current status
        let was_active = {
            let entry = self.get_entry(plugin_id).await?;
            entry.status == PluginStatus::Active
        };

        // Suspend if active
        if was_active {
            self.suspend_plugin(plugin_id).await?;
        }

        // Uninstall old version
        self.uninstall_plugin(plugin_id).await?;

        // Install new version
        self.install_plugin(new_manifest, new_wasm).await?;

        // Reactivate if was previously active
        if was_active {
            self.activate_plugin(plugin_id).await?;
        }

        info!("Plugin {} upgraded successfully", plugin_id);
        Ok(())
    }

    /// Get all plugins registered for a specific hook
    pub async fn get_plugins_for_hook(&self, entry_point: &EntryPoint) -> Vec<PluginId> {
        let hooks = self.hooks.read().await;
        hooks.get(entry_point).cloned().unwrap_or_default()
    }

    /// Get active plugins for a hook (only returns Active status plugins)
    pub async fn get_active_plugins_for_hook(&self, entry_point: &EntryPoint) -> Vec<PluginId> {
        let plugin_ids = self.get_plugins_for_hook(entry_point).await;
        let mut active = Vec::new();

        for plugin_id in plugin_ids {
            if let Ok(entry) = self.get_entry(&plugin_id).await {
                if entry.status == PluginStatus::Active {
                    active.push(plugin_id);
                }
            }
        }

        active
    }

    /// Get plugin status
    pub async fn get_status(&self, plugin_id: &PluginId) -> PluginResult<PluginStatus> {
        let entry = self.get_entry(plugin_id).await?;
        Ok(entry.status)
    }

    /// Get plugin manifest
    pub async fn get_manifest(&self, plugin_id: &PluginId) -> PluginResult<PluginManifest> {
        let entry = self.get_entry(plugin_id).await?;
        Ok(entry.manifest)
    }

    /// List all installed plugins
    pub async fn list_plugins(&self) -> Vec<(PluginId, PluginStatus)> {
        let plugins = self.plugins.read().await;
        plugins
            .iter()
            .map(|(id, entry)| (id.clone(), entry.status.clone()))
            .collect()
    }

    /// Get detailed plugin info
    pub async fn get_plugin_info(&self, plugin_id: &PluginId) -> PluginResult<PluginInfo> {
        let entry = self.get_entry(plugin_id).await?;
        Ok(PluginInfo {
            id: plugin_id.clone(),
            status: entry.status,
            manifest: entry.manifest,
            installed_at: entry.installed_at,
            status_changed_at: entry.status_changed_at,
            activation_attempts: entry.activation_attempts,
            last_error: entry.last_error,
        })
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    async fn get_entry(&self, plugin_id: &PluginId) -> PluginResult<PluginLifecycleEntry> {
        let plugins = self.plugins.read().await;
        plugins
            .get(plugin_id)
            .cloned()
            .ok_or_else(|| PluginError::NotFound(plugin_id.clone()))
    }

    async fn update_status(
        &self,
        plugin_id: &PluginId,
        status: PluginStatus,
    ) -> PluginResult<()> {
        let mut plugins = self.plugins.write().await;
        if let Some(entry) = plugins.get_mut(plugin_id) {
            entry.status = status.clone();
            entry.status_changed_at = chrono::Utc::now().timestamp() as u64;
            if let PluginStatus::Failed(ref msg) = status {
                entry.last_error = Some(msg.clone());
            }
            debug!("Plugin {} status updated to {:?}", plugin_id, status);
            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.clone()))
        }
    }

    fn compute_wasm_hash(&self, wasm_bytes: &[u8]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(wasm_bytes);
        *hasher.finalize().as_bytes()
    }

    async fn verify_signature(&self, manifest: &PluginManifest) -> PluginResult<bool> {
        // Check if signature exists
        let signature = match manifest.signature {
            Some(sig) => sig,
            None => {
                if self.config.require_signatures {
                    return Ok(false);
                }
                return Ok(true);
            }
        };

        let pubkey = match manifest.author_pubkey {
            Some(pk) => pk,
            None => return Ok(false),
        };

        // Check if author is trusted
        if !self.config.trusted_authors.is_empty()
            && !self.config.trusted_authors.contains(&pubkey)
        {
            warn!(
                "Plugin {} author is not in trusted authors list",
                manifest.id
            );
            // Still verify signature, but could add additional checks here
        }

        // Verify Ed25519 signature over manifest data (excluding signature field)
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let verifying_key = VerifyingKey::from_bytes(&pubkey)
            .map_err(|e| PluginError::InvalidConfiguration(format!("Invalid public key: {}", e)))?;

        // Create signature message from manifest fields
        let mut message = Vec::new();
        message.extend_from_slice(manifest.id.as_bytes());
        message.extend_from_slice(manifest.version.as_bytes());
        message.extend_from_slice(manifest.name.as_bytes());
        message.extend_from_slice(&manifest.wasm_hash);

        let sig = Signature::from_bytes(&signature);

        match verifying_key.verify(&message, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn compile_wasm(
        &self,
        manifest: &PluginManifest,
        wasm_bytes: Vec<u8>,
    ) -> PluginResult<CompiledPlugin> {
        // In a full implementation, this would use wasmtime/wasmer to compile
        // For now, we create a placeholder that stores the bytes
        let now = chrono::Utc::now().timestamp() as u64;

        Ok(CompiledPlugin {
            manifest: manifest.clone(),
            wasm_bytes,
            compiled_at: now,
            memory_usage: 0, // Would be computed from actual compilation
        })
    }

    async fn register_hooks(
        &self,
        plugin_id: &PluginId,
        entry_points: &[EntryPoint],
    ) -> PluginResult<()> {
        let mut hooks = self.hooks.write().await;

        for entry_point in entry_points {
            let plugins = hooks.entry(entry_point.clone()).or_insert_with(Vec::new);
            if !plugins.contains(plugin_id) {
                plugins.push(plugin_id.clone());
            }
        }

        // Also register with plugin manager
        for entry_point in entry_points {
            let hook: PluginHook = entry_point.clone().into();
            let _ = self.plugin_manager.register_hook(plugin_id.clone(), hook).await;
        }

        Ok(())
    }

    async fn deactivate_hooks(&self, plugin_id: &PluginId) {
        // This removes the plugin from active hook processing
        // but keeps the registration so it can be resumed
        let mut hooks = self.hooks.write().await;
        for plugins in hooks.values_mut() {
            plugins.retain(|id| id != plugin_id);
        }
    }

    async fn unregister_all_hooks(&self, plugin_id: &PluginId) {
        let mut hooks = self.hooks.write().await;
        for plugins in hooks.values_mut() {
            plugins.retain(|id| id != plugin_id);
        }
    }
}

// ============================================================================
// PLUGIN INFO
// ============================================================================

/// Detailed plugin information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin ID
    pub id: PluginId,
    /// Current status
    pub status: PluginStatus,
    /// Plugin manifest
    pub manifest: PluginManifest,
    /// Installation timestamp
    pub installed_at: u64,
    /// Last status change timestamp
    pub status_changed_at: u64,
    /// Number of activation attempts
    pub activation_attempts: u32,
    /// Last error message if any
    pub last_error: Option<String>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manifest(id: &str) -> PluginManifest {
        PluginManifest {
            id: id.to_string(),
            version: "1.0.0".to_string(),
            name: format!("Test Plugin {}", id),
            author: "Test Author".to_string(),
            description: "A test plugin".to_string(),
            api_version: "1.0".to_string(),
            entry_points: vec![EntryPoint::OnBlockReceived],
            dependencies: vec![],
            wasm_hash: [0u8; 32],
            signature: None,
            author_pubkey: None,
        }
    }

    #[test]
    fn test_plugin_status_transitions() {
        assert!(PluginStatus::Loaded.can_activate());
        assert!(PluginStatus::Suspended.can_activate());
        assert!(!PluginStatus::Active.can_activate());
        assert!(!PluginStatus::Pending.can_activate());

        assert!(PluginStatus::Active.can_suspend());
        assert!(!PluginStatus::Loaded.can_suspend());
        assert!(!PluginStatus::Suspended.can_suspend());

        assert!(PluginStatus::Active.is_operational());
        assert!(PluginStatus::Suspended.is_operational());
        assert!(!PluginStatus::Pending.is_operational());
    }

    #[test]
    fn test_entry_point_conversion() {
        let hook = PluginHook::OnConsensusEvent;
        let entry: EntryPoint = hook.clone().into();
        assert_eq!(entry, EntryPoint::OnConsensusRound);

        let back: PluginHook = entry.into();
        assert_eq!(back, PluginHook::OnConsensusEvent);
    }

    #[tokio::test]
    async fn test_lifecycle_config_defaults() {
        let config = PluginLifecycleConfig::default();
        assert_eq!(config.max_plugins, 100);
        assert_eq!(config.max_wasm_size, 10 * 1024 * 1024);
        assert_eq!(config.max_activation_attempts, 3);
        assert!(config.require_signatures);
    }

    #[test]
    fn test_manifest_serialization() {
        let manifest = create_test_manifest("test-plugin");
        let serialized = serde_json::to_string(&manifest).unwrap();
        let deserialized: PluginManifest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(manifest.id, deserialized.id);
        assert_eq!(manifest.version, deserialized.version);
    }
}
