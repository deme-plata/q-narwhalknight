//! Error types for the Q-Plugin framework
//!
//! Provides comprehensive error handling for plugin loading, validation,
//! execution, and lifecycle management.

use thiserror::Error;

/// Primary error type for the plugin framework
#[derive(Debug, Error)]
pub enum PluginError {
    // ============================================================================
    // MANIFEST ERRORS
    // ============================================================================
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    #[error("Manifest parsing failed: {0}")]
    ManifestParseError(String),

    #[error("Missing required manifest field: {0}")]
    MissingManifestField(String),

    #[error("Invalid plugin ID format: {0}")]
    InvalidPluginId(String),

    #[error("Invalid version format: {0}")]
    InvalidVersion(String),

    // ============================================================================
    // SIGNATURE AND VERIFICATION ERRORS
    // ============================================================================
    #[error("Signature verification failed: {0}")]
    SignatureVerificationFailed(String),

    #[error("Invalid signature format: {0}")]
    InvalidSignature(String),

    #[error("WASM hash mismatch: expected {expected}, got {actual}")]
    WasmHashMismatch { expected: String, actual: String },

    #[error("Author public key invalid: {0}")]
    InvalidAuthorKey(String),

    #[error("Plugin not signed by trusted author")]
    UntrustedAuthor,

    // ============================================================================
    // CAPABILITY ERRORS
    // ============================================================================
    #[error("Capability denied: {capability}")]
    CapabilityDenied { capability: String },

    #[error("Insufficient capabilities for operation: {operation}")]
    InsufficientCapabilities { operation: String },

    #[error("Invalid capability set: {0}")]
    InvalidCapabilitySet(String),

    #[error("Capability conflict: {0}")]
    CapabilityConflict(String),

    // ============================================================================
    // REGISTRY ERRORS
    // ============================================================================
    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    #[error("Plugin already registered: {0}")]
    PluginAlreadyRegistered(String),

    #[error("Plugin version conflict: {plugin_id} - existing: {existing}, new: {new}")]
    VersionConflict {
        plugin_id: String,
        existing: String,
        new: String,
    },

    #[error("Registry capacity exceeded: max {max}, attempted {attempted}")]
    RegistryCapacityExceeded { max: usize, attempted: usize },

    #[error("Plugin dependency not met: {plugin_id} requires {dependency}")]
    DependencyNotMet {
        plugin_id: String,
        dependency: String,
    },

    // ============================================================================
    // WASM RUNTIME ERRORS
    // ============================================================================
    #[error("WASM compilation failed: {0}")]
    WasmCompilationFailed(String),

    #[error("WASM instantiation failed: {0}")]
    WasmInstantiationFailed(String),

    #[error("WASM execution failed: {0}")]
    WasmExecutionFailed(String),

    #[error("WASM trap occurred: {0}")]
    WasmTrap(String),

    #[error("Invalid WASM bytecode: {0}")]
    InvalidWasmBytecode(String),

    #[error("WASM memory limit exceeded: limit {limit} bytes, requested {requested} bytes")]
    WasmMemoryLimitExceeded { limit: u64, requested: u64 },

    #[error("WASM fuel exhausted")]
    WasmFuelExhausted,

    // ============================================================================
    // EXECUTION ERRORS
    // ============================================================================
    #[error("Execution timeout: exceeded {limit_ms}ms")]
    ExecutionTimeout { limit_ms: u64 },

    #[error("Entry point not found: {0}")]
    EntryPointNotFound(String),

    #[error("Invalid entry point signature: {entry_point}")]
    InvalidEntryPointSignature { entry_point: String },

    #[error("Gas limit exceeded: limit {limit}, used {used}")]
    GasLimitExceeded { limit: u64, used: u64 },

    #[error("Plugin panicked: {0}")]
    PluginPanic(String),

    // ============================================================================
    // STATE ERRORS
    // ============================================================================
    #[error("State read error: {0}")]
    StateReadError(String),

    #[error("State write error: {0}")]
    StateWriteError(String),

    #[error("State access denied: plugin {plugin_id} cannot access {key}")]
    StateAccessDenied { plugin_id: String, key: String },

    #[error("State corruption detected: {0}")]
    StateCorruption(String),

    // ============================================================================
    // LIFECYCLE ERRORS
    // ============================================================================
    #[error("Plugin not initialized")]
    NotInitialized,

    #[error("Plugin already initialized")]
    AlreadyInitialized,

    #[error("Plugin shutdown failed: {0}")]
    ShutdownFailed(String),

    #[error("Plugin upgrade failed: {0}")]
    UpgradeFailed(String),

    // ============================================================================
    // I/O ERRORS
    // ============================================================================
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    // ============================================================================
    // INTERNAL ERRORS
    // ============================================================================
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Wasmtime error: {0}")]
    WasmtimeError(String),

    #[error("Lock acquisition failed")]
    LockError,
}

impl From<wasmtime::Error> for PluginError {
    fn from(err: wasmtime::Error) -> Self {
        PluginError::WasmtimeError(err.to_string())
    }
}

impl From<bincode::Error> for PluginError {
    fn from(err: bincode::Error) -> Self {
        PluginError::SerializationError(err.to_string())
    }
}

impl From<serde_json::Error> for PluginError {
    fn from(err: serde_json::Error) -> Self {
        PluginError::SerializationError(err.to_string())
    }
}

impl From<semver::Error> for PluginError {
    fn from(err: semver::Error) -> Self {
        PluginError::InvalidVersion(err.to_string())
    }
}

impl From<ed25519_dalek::SignatureError> for PluginError {
    fn from(err: ed25519_dalek::SignatureError) -> Self {
        PluginError::SignatureVerificationFailed(err.to_string())
    }
}

/// Result type alias for plugin operations
pub type PluginResult<T> = Result<T, PluginError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PluginError::PluginNotFound("test-plugin".to_string());
        assert_eq!(format!("{}", err), "Plugin not found: test-plugin");

        let err = PluginError::WasmHashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert!(format!("{}", err).contains("abc123"));
        assert!(format!("{}", err).contains("def456"));
    }

    #[test]
    fn test_error_conversions() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let plugin_err: PluginError = io_err.into();
        assert!(matches!(plugin_err, PluginError::IoError(_)));
    }
}
