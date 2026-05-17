//! Plugin registry for managing loaded plugins
//!
//! The registry maintains an in-memory index of all registered plugins,
//! handles version management, dependency resolution, and plugin lifecycle.

use crate::error::{PluginError, PluginResult};
use crate::manifest::{CapabilitySet, EntryPoint, PluginDependency, PluginId, PluginManifest};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

// ============================================================================
// PLUGIN ENTRY
// ============================================================================

/// A registered plugin entry in the registry
#[derive(Debug, Clone)]
pub struct PluginEntry {
    /// The plugin manifest
    pub manifest: PluginManifest,

    /// Plugin status
    pub status: PluginStatus,

    /// When the plugin was registered (Unix epoch seconds)
    pub registered_at: u64,

    /// When the plugin was last executed (Unix epoch seconds)
    pub last_executed_at: Option<u64>,

    /// Execution statistics
    pub stats: PluginStats,

    /// Compiled WASM module (stored separately for efficiency)
    pub wasm_bytecode: Option<Vec<u8>>,

    /// Runtime configuration overrides
    pub config_overrides: HashMap<String, String>,
}

impl PluginEntry {
    /// Create a new plugin entry from a manifest
    pub fn new(manifest: PluginManifest, wasm_bytecode: Option<Vec<u8>>) -> Self {
        Self {
            manifest,
            status: PluginStatus::Registered,
            registered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_executed_at: None,
            stats: PluginStats::default(),
            wasm_bytecode,
            config_overrides: HashMap::new(),
        }
    }

    /// Mark the plugin as executed
    pub fn mark_executed(&mut self, success: bool, gas_used: u64, execution_time_ms: u64) {
        self.last_executed_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
        self.stats.total_executions += 1;
        if success {
            self.stats.successful_executions += 1;
        } else {
            self.stats.failed_executions += 1;
        }
        self.stats.total_gas_used += gas_used;
        self.stats.total_execution_time_ms += execution_time_ms;
    }
}

/// Plugin execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginStatus {
    /// Plugin is registered but not yet initialized
    Registered,

    /// Plugin is initializing
    Initializing,

    /// Plugin is active and ready for execution
    Active,

    /// Plugin is paused (won't receive hooks)
    Paused,

    /// Plugin encountered an error
    Error,

    /// Plugin is being upgraded
    Upgrading,

    /// Plugin has been disabled
    Disabled,

    /// Plugin is being unregistered
    Unregistering,
}

impl std::fmt::Display for PluginStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginStatus::Registered => write!(f, "Registered"),
            PluginStatus::Initializing => write!(f, "Initializing"),
            PluginStatus::Active => write!(f, "Active"),
            PluginStatus::Paused => write!(f, "Paused"),
            PluginStatus::Error => write!(f, "Error"),
            PluginStatus::Upgrading => write!(f, "Upgrading"),
            PluginStatus::Disabled => write!(f, "Disabled"),
            PluginStatus::Unregistering => write!(f, "Unregistering"),
        }
    }
}

/// Plugin execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginStats {
    /// Total number of executions
    pub total_executions: u64,

    /// Number of successful executions
    pub successful_executions: u64,

    /// Number of failed executions
    pub failed_executions: u64,

    /// Total gas consumed
    pub total_gas_used: u64,

    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,

    /// Number of state reads
    pub state_reads: u64,

    /// Number of state writes
    pub state_writes: u64,

    /// Number of events emitted
    pub events_emitted: u64,
}

impl PluginStats {
    /// Calculate average execution time
    pub fn average_execution_time_ms(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.total_execution_time_ms as f64 / self.total_executions as f64
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            1.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }
}

// ============================================================================
// REGISTRY CONFIGURATION
// ============================================================================

/// Configuration for the plugin registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of plugins that can be registered
    pub max_plugins: usize,

    /// Maximum total memory for all plugins (bytes)
    pub max_total_memory: u64,

    /// Whether to allow unsigned plugins
    pub allow_unsigned: bool,

    /// Trusted author public keys
    pub trusted_authors: HashSet<[u8; 32]>,

    /// Whether to enforce capability restrictions
    pub enforce_capabilities: bool,

    /// Maximum allowed capabilities (ceiling for any plugin)
    pub max_capabilities: CapabilitySet,

    /// Auto-pause plugins after consecutive failures
    pub auto_pause_after_failures: u32,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_plugins: 100,
            max_total_memory: 1024 * 1024 * 1024, // 1GB total
            allow_unsigned: false,
            trusted_authors: HashSet::new(),
            enforce_capabilities: true,
            max_capabilities: CapabilitySet::full(),
            auto_pause_after_failures: 5,
        }
    }
}

impl RegistryConfig {
    /// Create a permissive config for development
    pub fn development() -> Self {
        Self {
            allow_unsigned: true,
            enforce_capabilities: false,
            ..Default::default()
        }
    }

    /// Create a strict config for production
    pub fn production() -> Self {
        Self {
            allow_unsigned: false,
            enforce_capabilities: true,
            max_plugins: 50,
            auto_pause_after_failures: 3,
            ..Default::default()
        }
    }
}

// ============================================================================
// PLUGIN REGISTRY
// ============================================================================

/// In-memory registry of loaded plugins
///
/// Thread-safe and supports concurrent access from multiple threads.
pub struct PluginRegistry {
    /// Configuration
    config: RwLock<RegistryConfig>,

    /// Registered plugins (plugin_id -> entry)
    plugins: DashMap<String, PluginEntry>,

    /// Index by entry point for fast lookup
    entry_point_index: DashMap<String, HashSet<String>>,

    /// Version history for each plugin (for upgrades)
    version_history: DashMap<String, Vec<semver::Version>>,

    /// Dependency graph (plugin_id -> dependencies)
    dependencies: DashMap<String, Vec<PluginDependency>>,

    /// Reverse dependency graph (plugin_id -> dependents)
    dependents: DashMap<String, HashSet<String>>,
}

impl PluginRegistry {
    /// Create a new plugin registry with default configuration
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create a new plugin registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        Self {
            config: RwLock::new(config),
            plugins: DashMap::new(),
            entry_point_index: DashMap::new(),
            version_history: DashMap::new(),
            dependencies: DashMap::new(),
            dependents: DashMap::new(),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> RegistryConfig {
        self.config.read().clone()
    }

    /// Update the configuration
    pub fn update_config(&self, config: RegistryConfig) {
        *self.config.write() = config;
    }

    /// Register a new plugin
    pub fn register(
        &self,
        manifest: PluginManifest,
        wasm_bytecode: Option<Vec<u8>>,
    ) -> PluginResult<()> {
        let config = self.config.read();
        let plugin_id = manifest.id.as_str().to_string();

        // Check capacity
        if self.plugins.len() >= config.max_plugins {
            return Err(PluginError::RegistryCapacityExceeded {
                max: config.max_plugins,
                attempted: self.plugins.len() + 1,
            });
        }

        // Check if already registered
        if self.plugins.contains_key(&plugin_id) {
            return Err(PluginError::PluginAlreadyRegistered(plugin_id));
        }

        // Verify signature if required
        if !config.allow_unsigned {
            if !manifest.verify_signature()? {
                return Err(PluginError::SignatureVerificationFailed(
                    "Invalid manifest signature".to_string(),
                ));
            }

            // Check trusted authors
            if !config.trusted_authors.is_empty()
                && !config.trusted_authors.contains(&manifest.author)
            {
                return Err(PluginError::UntrustedAuthor);
            }
        }

        // Verify WASM hash if bytecode provided
        if let Some(ref bytecode) = wasm_bytecode {
            manifest.verify_wasm_hash(bytecode)?;
        }

        // Check capabilities against max allowed
        if config.enforce_capabilities && !config.max_capabilities.satisfies(&manifest.capabilities)
        {
            return Err(PluginError::InsufficientCapabilities {
                operation: "Plugin requests capabilities beyond system maximum".to_string(),
            });
        }

        // Check dependencies
        for dep in &manifest.dependencies {
            if !dep.optional && !self.is_dependency_satisfied(dep)? {
                return Err(PluginError::DependencyNotMet {
                    plugin_id: plugin_id.clone(),
                    dependency: dep.plugin_id.as_str().to_string(),
                });
            }
        }

        drop(config); // Release read lock

        // Create entry
        let entry = PluginEntry::new(manifest.clone(), wasm_bytecode);

        // Update indices
        for entry_point in &manifest.entry_points {
            let key = entry_point.export_name();
            self.entry_point_index
                .entry(key)
                .or_insert_with(HashSet::new)
                .insert(plugin_id.clone());
        }

        // Update dependency graph
        self.dependencies
            .insert(plugin_id.clone(), manifest.dependencies.clone());
        for dep in &manifest.dependencies {
            self.dependents
                .entry(dep.plugin_id.as_str().to_string())
                .or_insert_with(HashSet::new)
                .insert(plugin_id.clone());
        }

        // Update version history
        self.version_history
            .entry(plugin_id.clone())
            .or_insert_with(Vec::new)
            .push(manifest.version.clone());

        // Insert plugin
        self.plugins.insert(plugin_id.clone(), entry);

        info!("Registered plugin: {} v{}", manifest.id, manifest.version);
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&self, plugin_id: &PluginId) -> PluginResult<PluginEntry> {
        let id = plugin_id.as_str();

        // Check if any plugins depend on this one
        if let Some(dependents) = self.dependents.get(id) {
            if !dependents.is_empty() {
                let dependent_list: Vec<_> = dependents.iter().cloned().collect();
                return Err(PluginError::CapabilityConflict(format!(
                    "Cannot unregister: plugins {:?} depend on {}",
                    dependent_list, id
                )));
            }
        }

        // Remove from main registry
        let (_, entry) = self
            .plugins
            .remove(id)
            .ok_or_else(|| PluginError::PluginNotFound(id.to_string()))?;

        // Remove from entry point index
        for entry_point in &entry.manifest.entry_points {
            let key = entry_point.export_name();
            if let Some(mut plugins) = self.entry_point_index.get_mut(&key) {
                plugins.remove(id);
            }
        }

        // Remove from dependency graph
        self.dependencies.remove(id);
        for dep in &entry.manifest.dependencies {
            if let Some(mut dependents) = self.dependents.get_mut(dep.plugin_id.as_str()) {
                dependents.remove(id);
            }
        }

        info!(
            "Unregistered plugin: {} v{}",
            entry.manifest.id, entry.manifest.version
        );
        Ok(entry)
    }

    /// Get a plugin by ID
    pub fn get(&self, plugin_id: &PluginId) -> Option<PluginEntry> {
        self.plugins.get(plugin_id.as_str()).map(|e| e.clone())
    }

    /// Get a plugin by ID string
    pub fn get_by_str(&self, plugin_id: &str) -> Option<PluginEntry> {
        self.plugins.get(plugin_id).map(|e| e.clone())
    }

    /// Check if a plugin is registered
    pub fn contains(&self, plugin_id: &PluginId) -> bool {
        self.plugins.contains_key(plugin_id.as_str())
    }

    /// List all registered plugins
    pub fn list(&self) -> Vec<PluginEntry> {
        self.plugins.iter().map(|e| e.value().clone()).collect()
    }

    /// List plugins that implement a specific entry point
    pub fn list_by_entry_point(&self, entry_point: &EntryPoint) -> Vec<PluginEntry> {
        let key = entry_point.export_name();
        if let Some(plugin_ids) = self.entry_point_index.get(&key) {
            plugin_ids
                .iter()
                .filter_map(|id| self.plugins.get(id).map(|e| e.value().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// List plugins with a specific status
    pub fn list_by_status(&self, status: PluginStatus) -> Vec<PluginEntry> {
        self.plugins
            .iter()
            .filter(|e| e.status == status)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Get the number of registered plugins
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Update plugin status
    pub fn update_status(&self, plugin_id: &PluginId, status: PluginStatus) -> PluginResult<()> {
        let mut entry = self
            .plugins
            .get_mut(plugin_id.as_str())
            .ok_or_else(|| PluginError::PluginNotFound(plugin_id.as_str().to_string()))?;

        let old_status = entry.status;
        entry.status = status;

        debug!(
            "Plugin {} status changed: {} -> {}",
            plugin_id, old_status, status
        );
        Ok(())
    }

    /// Record an execution for a plugin
    pub fn record_execution(
        &self,
        plugin_id: &PluginId,
        success: bool,
        gas_used: u64,
        execution_time_ms: u64,
    ) -> PluginResult<()> {
        let mut entry = self
            .plugins
            .get_mut(plugin_id.as_str())
            .ok_or_else(|| PluginError::PluginNotFound(plugin_id.as_str().to_string()))?;

        entry.mark_executed(success, gas_used, execution_time_ms);

        // Auto-pause on consecutive failures
        let config = self.config.read();
        if !success && config.auto_pause_after_failures > 0 {
            let consecutive_failures =
                entry.stats.total_executions - entry.stats.successful_executions;
            if consecutive_failures >= config.auto_pause_after_failures as u64 {
                warn!(
                    "Plugin {} paused after {} consecutive failures",
                    plugin_id, consecutive_failures
                );
                entry.status = PluginStatus::Paused;
            }
        }

        Ok(())
    }

    /// Upgrade a plugin to a new version
    pub fn upgrade(
        &self,
        new_manifest: PluginManifest,
        new_wasm_bytecode: Option<Vec<u8>>,
    ) -> PluginResult<semver::Version> {
        let plugin_id = new_manifest.id.as_str();

        // Get existing entry
        let existing = self
            .plugins
            .get(plugin_id)
            .ok_or_else(|| PluginError::PluginNotFound(plugin_id.to_string()))?;

        let old_version = existing.manifest.version.clone();

        // Check version is newer
        if new_manifest.version <= old_version {
            return Err(PluginError::VersionConflict {
                plugin_id: plugin_id.to_string(),
                existing: old_version.to_string(),
                new: new_manifest.version.to_string(),
            });
        }

        drop(existing); // Release the reference

        // Mark as upgrading
        self.update_status(&new_manifest.id, PluginStatus::Upgrading)?;

        // Unregister old version (this will fail if there are dependents with incompatible versions)
        let old_entry = self.unregister(&new_manifest.id)?;

        // Register new version
        match self.register(new_manifest.clone(), new_wasm_bytecode) {
            Ok(()) => {
                info!(
                    "Upgraded plugin {} from v{} to v{}",
                    new_manifest.id, old_version, new_manifest.version
                );
                Ok(old_version)
            }
            Err(e) => {
                // Rollback: re-register old version
                warn!("Upgrade failed, rolling back: {}", e);
                self.plugins.insert(
                    old_entry.manifest.id.as_str().to_string(),
                    old_entry.clone(),
                );
                Err(PluginError::UpgradeFailed(e.to_string()))
            }
        }
    }

    /// Check if a dependency is satisfied
    fn is_dependency_satisfied(&self, dep: &PluginDependency) -> PluginResult<bool> {
        if let Some(entry) = self.plugins.get(dep.plugin_id.as_str()) {
            dep.is_satisfied_by(&entry.manifest.version)
        } else {
            Ok(false)
        }
    }

    /// Get all plugins that depend on a given plugin
    pub fn get_dependents(&self, plugin_id: &PluginId) -> Vec<PluginId> {
        if let Some(dependents) = self.dependents.get(plugin_id.as_str()) {
            dependents
                .iter()
                .filter_map(|id| PluginId::new(id.clone()).ok())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get version history for a plugin
    pub fn get_version_history(&self, plugin_id: &PluginId) -> Vec<semver::Version> {
        self.version_history
            .get(plugin_id.as_str())
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get aggregate statistics for all plugins
    pub fn aggregate_stats(&self) -> AggregateStats {
        let mut stats = AggregateStats::default();

        for entry in self.plugins.iter() {
            stats.total_plugins += 1;
            match entry.status {
                PluginStatus::Active => stats.active_plugins += 1,
                PluginStatus::Paused => stats.paused_plugins += 1,
                PluginStatus::Error => stats.error_plugins += 1,
                _ => {}
            }
            stats.total_executions += entry.stats.total_executions;
            stats.total_gas_used += entry.stats.total_gas_used;
            stats.successful_executions += entry.stats.successful_executions;
            stats.failed_executions += entry.stats.failed_executions;
        }

        stats
    }

    /// Clear all plugins (for testing)
    #[cfg(test)]
    pub fn clear(&self) {
        self.plugins.clear();
        self.entry_point_index.clear();
        self.version_history.clear();
        self.dependencies.clear();
        self.dependents.clear();
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate statistics for the registry
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateStats {
    pub total_plugins: u64,
    pub active_plugins: u64,
    pub paused_plugins: u64,
    pub error_plugins: u64,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub total_gas_used: u64,
}

/// Thread-safe reference to a plugin registry
pub type SharedRegistry = Arc<PluginRegistry>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::CapabilitySet;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use sha3::{Digest, Sha3_256};

    fn create_test_manifest(id: &str, version: &str) -> PluginManifest {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let wasm_bytecode = b"(module)";
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytecode);
        let result = hasher.finalize();
        let mut wasm_hash = [0u8; 32];
        wasm_hash.copy_from_slice(&result);

        PluginManifest::builder()
            .id(PluginId::new(id).unwrap())
            .name(format!("Test Plugin {}", id))
            .version(semver::Version::parse(version).unwrap())
            .author(verifying_key.to_bytes())
            .wasm_hash(wasm_hash)
            .capabilities(CapabilitySet::minimal())
            .entry_point(EntryPoint::OnInit)
            .min_gas_limit(1000)
            .build_signed(&signing_key)
            .unwrap()
    }

    #[test]
    fn test_register_and_get() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());
        let manifest = create_test_manifest("test.plugin", "1.0.0");

        registry.register(manifest.clone(), None).unwrap();

        let entry = registry.get(&manifest.id).unwrap();
        assert_eq!(entry.manifest.id, manifest.id);
        assert_eq!(entry.status, PluginStatus::Registered);
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());
        let manifest = create_test_manifest("test.plugin", "1.0.0");

        registry.register(manifest.clone(), None).unwrap();
        let result = registry.register(manifest.clone(), None);

        assert!(matches!(result, Err(PluginError::PluginAlreadyRegistered(_))));
    }

    #[test]
    fn test_unregister() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());
        let manifest = create_test_manifest("test.plugin", "1.0.0");

        registry.register(manifest.clone(), None).unwrap();
        assert!(registry.contains(&manifest.id));

        registry.unregister(&manifest.id).unwrap();
        assert!(!registry.contains(&manifest.id));
    }

    #[test]
    fn test_upgrade() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());

        let manifest_v1 = create_test_manifest("test.plugin", "1.0.0");
        registry.register(manifest_v1.clone(), None).unwrap();

        let manifest_v2 = create_test_manifest("test.plugin", "2.0.0");
        let old_version = registry.upgrade(manifest_v2.clone(), None).unwrap();

        assert_eq!(old_version, semver::Version::new(1, 0, 0));

        let entry = registry.get(&manifest_v2.id).unwrap();
        assert_eq!(entry.manifest.version, semver::Version::new(2, 0, 0));
    }

    #[test]
    fn test_downgrade_rejected() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());

        let manifest_v2 = create_test_manifest("test.plugin", "2.0.0");
        registry.register(manifest_v2.clone(), None).unwrap();

        let manifest_v1 = create_test_manifest("test.plugin", "1.0.0");
        let result = registry.upgrade(manifest_v1, None);

        assert!(matches!(result, Err(PluginError::VersionConflict { .. })));
    }

    #[test]
    fn test_list_by_entry_point() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());

        let manifest1 = create_test_manifest("test.plugin1", "1.0.0");
        let manifest2 = create_test_manifest("test.plugin2", "1.0.0");

        registry.register(manifest1, None).unwrap();
        registry.register(manifest2, None).unwrap();

        let plugins = registry.list_by_entry_point(&EntryPoint::OnInit);
        assert_eq!(plugins.len(), 2);
    }

    #[test]
    fn test_execution_tracking() {
        let registry = PluginRegistry::with_config(RegistryConfig::development());
        let manifest = create_test_manifest("test.plugin", "1.0.0");

        registry.register(manifest.clone(), None).unwrap();

        registry
            .record_execution(&manifest.id, true, 1000, 50)
            .unwrap();
        registry
            .record_execution(&manifest.id, false, 500, 25)
            .unwrap();

        let entry = registry.get(&manifest.id).unwrap();
        assert_eq!(entry.stats.total_executions, 2);
        assert_eq!(entry.stats.successful_executions, 1);
        assert_eq!(entry.stats.failed_executions, 1);
        assert_eq!(entry.stats.total_gas_used, 1500);
    }

    #[test]
    fn test_capacity_limit() {
        let config = RegistryConfig {
            max_plugins: 2,
            allow_unsigned: true,
            ..Default::default()
        };
        let registry = PluginRegistry::with_config(config);

        registry
            .register(create_test_manifest("test.plugin1", "1.0.0"), None)
            .unwrap();
        registry
            .register(create_test_manifest("test.plugin2", "1.0.0"), None)
            .unwrap();

        let result = registry.register(create_test_manifest("test.plugin3", "1.0.0"), None);
        assert!(matches!(
            result,
            Err(PluginError::RegistryCapacityExceeded { .. })
        ));
    }
}
