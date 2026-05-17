//! Q-NarwhalKnight Plugin System
//!
//! A secure WASM plugin framework with capability-based permissions, gas metering,
//! and deterministic execution for the Q-NarwhalKnight quantum consensus system.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Plugin System                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  PluginExecutor                                                  │
//! │  ├── Engine (wasmtime, deterministic config)                    │
//! │  ├── Linker (host function bindings)                            │
//! │  └── Instances (loaded plugins with state)                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Host Functions                                                  │
//! │  ├── Storage (namespaced read/write)                            │
//! │  ├── Events (emit to blockchain)                                │
//! │  ├── Crypto (sha3, ed25519 verify)                              │
//! │  └── System (block height, timestamp, logging)                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Gas Metering                                                    │
//! │  ├── Fuel tracking (wasmtime fuel)                              │
//! │  ├── Per-operation costs                                         │
//! │  └── Out-of-gas handling                                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Deterministic execution**: Reproducible results across all nodes
//! - **Capability-based security**: Fine-grained permission control
//! - **Gas metering**: Resource limiting with wasmtime fuel
//! - **Namespaced storage**: Isolated storage per plugin
//! - **Event emission**: Plugins can emit events to the blockchain
//!
//! # Example
//!
//! ```rust,ignore
//! use q_plugin::{PluginExecutor, PluginManifest, Capability, CapabilitySet};
//!
//! // Create executor
//! let executor = PluginExecutor::new()?;
//!
//! // Define plugin manifest
//! let manifest = PluginManifest {
//!     id: "my-plugin".to_string(),
//!     name: "My Plugin".to_string(),
//!     version: semver::Version::new(1, 0, 0),
//!     author: "Developer".to_string(),
//!     description: "A sample plugin".to_string(),
//!     entry_points: vec!["process".to_string()],
//!     capabilities: CapabilitySet::new(&[
//!         Capability::StorageRead,
//!         Capability::StorageWrite,
//!         Capability::EmitEvent,
//!     ]),
//!     max_gas: 1_000_000,
//! };
//!
//! // Load and execute
//! let plugin_id = executor.load_plugin(&wasm_bytes, manifest)?;
//! let result = executor.execute(&plugin_id, "process", &input_data, 100_000)?;
//! ```

pub mod runtime;

// Re-export main types
pub use runtime::{
    executor::{PluginExecutor, PluginInstance, PluginState},
    gas::{GasCosts, GasMeter, GasMetrics},
    host_functions::HostFunctionRegistry,
};

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Plugin identifier type
pub type PluginId = String;

/// Plugin manifest describing a WASM plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin identifier
    pub id: PluginId,
    /// Human-readable name
    pub name: String,
    /// Semantic version
    pub version: semver::Version,
    /// Plugin author
    pub author: String,
    /// Description
    pub description: String,
    /// List of exported entry points
    pub entry_points: Vec<String>,
    /// Required capabilities
    pub capabilities: CapabilitySet,
    /// Maximum gas allowed per execution
    pub max_gas: u64,
}

impl PluginManifest {
    /// Create a new plugin manifest
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: semver::Version::new(1, 0, 0),
            author: String::new(),
            description: String::new(),
            entry_points: Vec::new(),
            capabilities: CapabilitySet::default(),
            max_gas: 1_000_000, // 1M gas default
        }
    }

    /// Set version
    pub fn with_version(mut self, major: u64, minor: u64, patch: u64) -> Self {
        self.version = semver::Version::new(major, minor, patch);
        self
    }

    /// Set author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add entry point
    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_points.push(entry_point.into());
        self
    }

    /// Set capabilities
    pub fn with_capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set max gas
    pub fn with_max_gas(mut self, max_gas: u64) -> Self {
        self.max_gas = max_gas;
        self
    }
}

/// Plugin capabilities for security
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Read from plugin's namespaced storage
    StorageRead,
    /// Write to plugin's namespaced storage
    StorageWrite,
    /// Emit events to the blockchain
    EmitEvent,
    /// Access current block height
    GetBlockHeight,
    /// Access current timestamp
    GetTimestamp,
    /// Verify Ed25519 signatures
    VerifySignature,
    /// Compute SHA3-256 hashes
    ComputeHash,
    /// Write log messages
    Log,
    /// Access network information (limited)
    NetworkInfo,
    /// Interact with other contracts (advanced)
    CrossContractCall,
}

impl Capability {
    /// Get all available capabilities
    pub fn all() -> Vec<Capability> {
        vec![
            Capability::StorageRead,
            Capability::StorageWrite,
            Capability::EmitEvent,
            Capability::GetBlockHeight,
            Capability::GetTimestamp,
            Capability::VerifySignature,
            Capability::ComputeHash,
            Capability::Log,
            Capability::NetworkInfo,
            Capability::CrossContractCall,
        ]
    }

    /// Get basic/safe capabilities
    pub fn basic() -> Vec<Capability> {
        vec![
            Capability::StorageRead,
            Capability::GetBlockHeight,
            Capability::GetTimestamp,
            Capability::ComputeHash,
            Capability::Log,
        ]
    }
}

/// Set of capabilities granted to a plugin
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitySet {
    capabilities: HashSet<Capability>,
}

impl CapabilitySet {
    /// Create a new capability set from a slice
    pub fn new(capabilities: &[Capability]) -> Self {
        Self {
            capabilities: capabilities.iter().copied().collect(),
        }
    }

    /// Create an empty capability set
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a set with all capabilities
    pub fn all() -> Self {
        Self::new(&Capability::all())
    }

    /// Create a set with basic safe capabilities
    pub fn basic() -> Self {
        Self::new(&Capability::basic())
    }

    /// Check if a capability is granted
    pub fn has(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// Add a capability
    pub fn grant(&mut self, capability: Capability) {
        self.capabilities.insert(capability);
    }

    /// Remove a capability
    pub fn revoke(&mut self, capability: Capability) {
        self.capabilities.remove(&capability);
    }

    /// Get the number of capabilities
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }

    /// Iterate over capabilities
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.iter()
    }
}

/// Plugin event emitted during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEvent {
    /// Event topic/name
    pub topic: String,
    /// Event data
    pub data: Vec<u8>,
    /// Timestamp when event was emitted
    pub timestamp: u64,
    /// Plugin that emitted the event
    pub plugin_id: PluginId,
}

impl PluginEvent {
    /// Create a new plugin event
    pub fn new(topic: impl Into<String>, data: Vec<u8>, plugin_id: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            plugin_id: plugin_id.into(),
        }
    }
}

/// Namespaced storage for plugins
#[derive(Debug, Clone, Default)]
pub struct PluginStorage {
    /// Storage data keyed by namespaced key
    data: std::collections::HashMap<String, Vec<u8>>,
    /// Plugin namespace prefix
    namespace: String,
}

impl PluginStorage {
    /// Create new storage with namespace
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            data: std::collections::HashMap::new(),
            namespace: namespace.into(),
        }
    }

    /// Get namespaced key
    fn namespaced_key(&self, key: &str) -> String {
        format!("{}:{}", self.namespace, key)
    }

    /// Read from storage
    pub fn read(&self, key: &str) -> Option<Vec<u8>> {
        let ns_key = self.namespaced_key(key);
        self.data.get(&ns_key).cloned()
    }

    /// Write to storage
    pub fn write(&mut self, key: &str, value: Vec<u8>) {
        let ns_key = self.namespaced_key(key);
        self.data.insert(ns_key, value);
    }

    /// Delete from storage
    pub fn delete(&mut self, key: &str) -> Option<Vec<u8>> {
        let ns_key = self.namespaced_key(key);
        self.data.remove(&ns_key)
    }

    /// Check if key exists
    pub fn exists(&self, key: &str) -> bool {
        let ns_key = self.namespaced_key(key);
        self.data.contains_key(&ns_key)
    }

    /// Get all keys (without namespace prefix)
    pub fn keys(&self) -> Vec<String> {
        let prefix = format!("{}:", self.namespace);
        self.data
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix).map(String::from))
            .collect()
    }

    /// Clear all storage
    pub fn clear(&mut self) {
        let prefix = format!("{}:", self.namespace);
        self.data.retain(|k, _| !k.starts_with(&prefix));
    }

    /// Get storage size in bytes
    pub fn size(&self) -> usize {
        let prefix = format!("{}:", self.namespace);
        self.data
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| k.len() + v.len())
            .sum()
    }
}

/// Execution result from plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Output data
    pub output: Vec<u8>,
    /// Gas consumed
    pub gas_used: u64,
    /// Events emitted
    pub events: Vec<PluginEvent>,
    /// Execution successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ExecutionResult {
    /// Create a successful result
    pub fn success(output: Vec<u8>, gas_used: u64, events: Vec<PluginEvent>) -> Self {
        Self {
            output,
            gas_used,
            events,
            success: true,
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(error: impl Into<String>, gas_used: u64) -> Self {
        Self {
            output: Vec::new(),
            gas_used,
            events: Vec::new(),
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Plugin system errors
#[derive(Debug, Error)]
pub enum PluginError {
    /// WASM compilation error
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// WASM instantiation error
    #[error("Instantiation error: {0}")]
    Instantiation(String),

    /// Runtime execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Plugin not found
    #[error("Plugin not found: {0}")]
    NotFound(String),

    /// Plugin already exists
    #[error("Plugin already exists: {0}")]
    AlreadyExists(String),

    /// Capability not granted
    #[error("Capability not granted: {0:?}")]
    CapabilityDenied(Capability),

    /// Out of gas
    #[error("Out of gas: used {used}, limit {limit}")]
    OutOfGas { used: u64, limit: u64 },

    /// Invalid entry point
    #[error("Invalid entry point: {0}")]
    InvalidEntryPoint(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid WASM module
    #[error("Invalid WASM module: {0}")]
    InvalidModule(String),

    /// Memory error
    #[error("Memory error: {0}")]
    Memory(String),

    /// Host function error
    #[error("Host function error: {0}")]
    HostFunction(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<wasmtime::Error> for PluginError {
    fn from(err: wasmtime::Error) -> Self {
        PluginError::Execution(err.to_string())
    }
}

impl From<bincode::Error> for PluginError {
    fn from(err: bincode::Error) -> Self {
        PluginError::Serialization(err.to_string())
    }
}

impl From<serde_json::Error> for PluginError {
    fn from(err: serde_json::Error) -> Self {
        PluginError::Serialization(err.to_string())
    }
}

/// Result type for plugin operations
pub type PluginResult<T> = Result<T, PluginError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_set() {
        let mut caps = CapabilitySet::empty();
        assert!(!caps.has(Capability::StorageRead));

        caps.grant(Capability::StorageRead);
        assert!(caps.has(Capability::StorageRead));

        caps.revoke(Capability::StorageRead);
        assert!(!caps.has(Capability::StorageRead));
    }

    #[test]
    fn test_plugin_storage() {
        let mut storage = PluginStorage::new("test-plugin");

        storage.write("key1", b"value1".to_vec());
        assert_eq!(storage.read("key1"), Some(b"value1".to_vec()));
        assert!(storage.exists("key1"));

        storage.delete("key1");
        assert!(!storage.exists("key1"));
    }

    #[test]
    fn test_plugin_manifest_builder() {
        let manifest = PluginManifest::new("test", "Test Plugin")
            .with_version(1, 2, 3)
            .with_author("Developer")
            .with_description("A test plugin")
            .with_entry_point("main")
            .with_capabilities(CapabilitySet::basic())
            .with_max_gas(500_000);

        assert_eq!(manifest.id, "test");
        assert_eq!(manifest.version, semver::Version::new(1, 2, 3));
        assert_eq!(manifest.author, "Developer");
        assert!(manifest.capabilities.has(Capability::StorageRead));
    }

    #[test]
    fn test_execution_result() {
        let success = ExecutionResult::success(b"output".to_vec(), 1000, vec![]);
        assert!(success.success);
        assert!(success.error.is_none());

        let failure = ExecutionResult::failure("error", 500);
        assert!(!failure.success);
        assert_eq!(failure.error, Some("error".to_string()));
    }
}
