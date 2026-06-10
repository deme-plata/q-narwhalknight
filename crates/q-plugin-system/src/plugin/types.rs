/// Plugin type definitions for Q-NarwhalKnight Plugin System
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export commonly used types
pub use crate::plugin::{PluginConfig, PluginId, PluginPermissions, PluginResourceLimits};

/// Plugin type enumeration for different execution environments
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginType {
    /// WebAssembly plugin
    Wasm,
    /// Native shared library
    Native,
    /// JavaScript plugin (for web environments)
    JavaScript,
    /// Python plugin (interpreted)
    Python,
    /// Custom plugin type
    Custom(String),
}

/// Plugin execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PluginPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for PluginPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub plugin_id: PluginId,
    pub version_requirement: String,
    pub optional: bool,
}

/// Extended plugin configuration with Q-NarwhalKnight specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedPluginConfig {
    pub basic_config: PluginConfig,
    pub plugin_type: PluginType,
    pub priority: PluginPriority,
    pub dependencies: Vec<PluginDependency>,
    pub quantum_features: QuantumFeatures,
    pub consensus_features: ConsensusFeatures,
    pub network_features: NetworkFeatures,
}

/// Quantum-specific plugin features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatures {
    pub quantum_random_access: bool,
    pub quantum_crypto_support: bool,
    pub quantum_state_interaction: bool,
    pub quantum_entanglement_protocols: bool,
}

impl Default for QuantumFeatures {
    fn default() -> Self {
        Self {
            quantum_random_access: false,
            quantum_crypto_support: false,
            quantum_state_interaction: false,
            quantum_entanglement_protocols: false,
        }
    }
}

/// Consensus-specific plugin features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusFeatures {
    pub consensus_participation: bool,
    pub validator_operations: bool,
    pub block_production: bool,
    pub byzantine_fault_tolerance: bool,
    pub dag_operations: bool,
}

impl Default for ConsensusFeatures {
    fn default() -> Self {
        Self {
            consensus_participation: false,
            validator_operations: false,
            block_production: false,
            byzantine_fault_tolerance: false,
            dag_operations: false,
        }
    }
}

/// Network-specific plugin features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkFeatures {
    pub tor_integration: bool,
    pub libp2p_protocols: bool,
    pub quantum_network_protocols: bool,
    pub mesh_networking: bool,
    pub anonymity_protocols: bool,
}

impl Default for NetworkFeatures {
    fn default() -> Self {
        Self {
            tor_integration: false,
            libp2p_protocols: false,
            quantum_network_protocols: false,
            mesh_networking: false,
            anonymity_protocols: false,
        }
    }
}

/// Plugin execution environment context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEnvironment {
    pub runtime_version: String,
    pub available_memory: u64,
    pub available_storage: u64,
    pub network_connectivity: bool,
    pub quantum_hardware_access: bool,
    pub consensus_node_type: ConsensusNodeType,
}

/// Type of consensus node for plugin context
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusNodeType {
    Validator,
    FullNode,
    LightClient,
    Relayer,
}

/// Plugin capability declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    pub supported_hooks: Vec<crate::plugin::PluginHook>,
    pub provides_services: Vec<String>,
    pub consumes_services: Vec<String>,
    pub api_version: String,
    pub feature_flags: HashMap<String, bool>,
}

/// Plugin metadata for discovery and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub id: PluginId,
    pub config: ExtendedPluginConfig,
    pub capabilities: PluginCapabilities,
    pub environment: PluginEnvironment,
    pub installation_path: String,
    pub checksum: String,
    pub signature: Option<String>,
}

/// Plugin installation package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginPackage {
    pub metadata: PluginMetadata,
    pub binary_data: Vec<u8>,
    pub resources: HashMap<String, Vec<u8>>,
    pub configuration_schema: Option<String>,
}

/// Plugin registry entry for distributed plugin discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistryEntry {
    pub metadata: PluginMetadata,
    pub download_url: String,
    pub publisher_key: String,
    pub publication_date: u64,
    pub verification_status: VerificationStatus,
}

/// Plugin verification status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Unverified,
    SignatureValid,
    PublisherVerified,
    OfficiallySupported,
}

/// Plugin execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time_ms: f64,
    pub total_gas_consumed: u64,
    pub memory_peak_usage: u64,
    pub network_bytes_transferred: u64,
}

/// Plugin health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginHealth {
    Healthy,
    Warning(String),
    Critical(String),
    Failed(String),
}

/// Plugin service interface for inter-plugin communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginService {
    pub service_name: String,
    pub service_version: String,
    pub endpoints: Vec<ServiceEndpoint>,
    pub provider_plugin: PluginId,
}

/// Service endpoint for plugin services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub name: String,
    pub method: String,
    pub input_schema: String,
    pub output_schema: String,
    pub gas_cost: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_type_serialization() {
        let wasm_type = PluginType::Wasm;
        let serialized = serde_json::to_string(&wasm_type).unwrap();
        let deserialized: PluginType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(wasm_type, deserialized);
    }

    #[test]
    fn test_plugin_priority_ordering() {
        assert!(PluginPriority::Critical > PluginPriority::High);
        assert!(PluginPriority::High > PluginPriority::Normal);
        assert!(PluginPriority::Normal > PluginPriority::Low);
    }

    #[test]
    fn test_extended_config_defaults() {
        let quantum_features = QuantumFeatures::default();
        assert!(!quantum_features.quantum_random_access);
        assert!(!quantum_features.quantum_crypto_support);
    }
}
