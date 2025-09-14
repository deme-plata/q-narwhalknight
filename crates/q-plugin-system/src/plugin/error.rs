use serde::{Deserialize, Serialize};
/// Plugin system error types for Q-NarwhalKnight
use thiserror::Error;

/// Comprehensive error types for plugin operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),

    #[error("Plugin already exists: {0}")]
    AlreadyExists(String),

    #[error("Plugin execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    #[error("Invalid plugin configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Plugin load failed: {0}")]
    LoadFailed(String),

    #[error("Plugin unload failed: {0}")]
    UnloadFailed(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timeout occurred: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("VM integration error: {0}")]
    VmIntegrationError(String),

    #[error("Consensus integration error: {0}")]
    ConsensusError(String),

    #[error("Transaction processing error: {0}")]
    TransactionError(String),
}

impl PluginError {
    /// Create a not found error
    pub fn not_found(plugin_id: &str) -> Self {
        Self::NotFound(plugin_id.to_string())
    }

    /// Create a permission denied error
    pub fn permission_denied(message: &str) -> Self {
        Self::PermissionDenied(message.to_string())
    }

    /// Create a resource limit exceeded error
    pub fn resource_limit_exceeded(message: &str) -> Self {
        Self::ResourceLimitExceeded(message.to_string())
    }

    /// Create an execution failed error
    pub fn execution_failed(message: &str) -> Self {
        Self::ExecutionFailed(message.to_string())
    }

    /// Create an internal error
    pub fn internal(message: &str) -> Self {
        Self::Internal(message.to_string())
    }
}

/// Specialized error types for different plugin system components
#[derive(Debug, Error)]
pub enum PluginVmError {
    #[error("VM bridge error: {0}")]
    BridgeError(String),

    #[error("State access error: {0}")]
    StateAccessError(String),

    #[error("Gas limit exceeded")]
    GasLimitExceeded,

    #[error("Invalid call data: {0}")]
    InvalidCallData(String),
}

#[derive(Debug, Error)]
pub enum PluginConsensusError {
    #[error("Consensus integration error: {0}")]
    IntegrationError(String),

    #[error("Invalid consensus operation: {0}")]
    InvalidOperation(String),

    #[error("Consensus phase error: {0}")]
    PhaseError(String),
}

#[derive(Debug, Error)]
pub enum PluginTransactionError {
    #[error("Transaction validation failed: {0}")]
    ValidationFailed(String),

    #[error("Transaction execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid transaction type: {0}")]
    InvalidType(String),

    #[error("Transaction rollback failed: {0}")]
    RollbackFailed(String),
}

/// Convert from other error types
impl From<bincode::Error> for PluginError {
    fn from(err: bincode::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

impl From<serde_json::Error> for PluginError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for PluginError {
    fn from(err: std::io::Error) -> Self {
        Self::Internal(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for PluginError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        Self::Timeout("Plugin execution timed out".to_string())
    }
}
