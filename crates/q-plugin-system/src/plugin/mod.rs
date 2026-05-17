pub mod error;
/// Plugin management and execution framework for Q-NarwhalKnight
///
/// Core plugin types and management infrastructure
pub mod manager;
pub mod types;

pub use error::*;
pub use manager::*;
pub use types::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Plugin identifier type
pub type PluginId = String;

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub entry_point: String,
    pub permissions: PluginPermissions,
    pub resource_limits: PluginResourceLimits,
}

/// Plugin permissions within the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginPermissions {
    pub network_access: bool,
    pub file_system_access: bool,
    pub consensus_participation: bool,
    pub state_modification: bool,
    pub transaction_processing: bool,
}

impl Default for PluginPermissions {
    fn default() -> Self {
        Self {
            network_access: false,
            file_system_access: false,
            consensus_participation: false,
            state_modification: false,
            transaction_processing: false,
        }
    }
}

/// Plugin resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResourceLimits {
    pub max_memory_bytes: u64,
    pub max_execution_time_ms: u64,
    pub max_network_calls_per_second: u32,
    pub max_storage_bytes: u64,
}

impl Default for PluginResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 50_000_000, // 50MB
            max_execution_time_ms: 1000,  // 1 second
            max_network_calls_per_second: 10,
            max_storage_bytes: 10_000_000, // 10MB
        }
    }
}

/// Plugin execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginState {
    pub storage: HashMap<String, Vec<u8>>,
    pub resource_usage: ResourceUsage,
    pub network_stats: NetworkStats,
}

/// Resource usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_used: u64,
    pub execution_time_ms: u64,
    pub network_calls_made: u32,
    pub storage_used: u64,
}

/// Network statistics for plugins
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connections_opened: u32,
    pub requests_made: u32,
}

/// Plugin execution context
pub struct PluginExecutionContext {
    pub plugin_id: PluginId,
    pub execution_timestamp: u64,
    pub available_resources: PluginResourceLimits,
    pub permissions: PluginPermissions,
}

/// Plugin hook types for lifecycle management
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginHook {
    OnLoad,
    OnUnload,
    BeforeTransaction,
    AfterTransaction,
    BeforeBlock,
    AfterBlock,
    OnConsensusEvent,
    OnNetworkEvent,
    OnStateChange,
    Custom(String),
}

/// Plugin execution result
pub type PluginResult<T> = Result<T, PluginError>;
