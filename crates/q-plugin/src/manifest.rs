//! Plugin manifest types for Q-NarwhalKnight
//!
//! The manifest defines a plugin's identity, capabilities, entry points,
//! and resource requirements. All plugins must provide a signed manifest
//! to ensure authenticity and establish trust.

use crate::error::{PluginError, PluginResult};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

// ============================================================================
// PLUGIN ID
// ============================================================================

/// Unique identifier for a plugin
///
/// Plugin IDs follow the format: `namespace.name` (e.g., `qnk.validator`, `community.dex`)
/// - Namespace: lowercase alphanumeric with hyphens, 2-32 chars
/// - Name: lowercase alphanumeric with hyphens, 2-64 chars
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginId(String);

impl PluginId {
    /// Create a new plugin ID with validation
    pub fn new(id: impl Into<String>) -> PluginResult<Self> {
        let id = id.into();
        Self::validate(&id)?;
        Ok(PluginId(id))
    }

    /// Create a plugin ID without validation (for internal use)
    pub(crate) fn new_unchecked(id: String) -> Self {
        PluginId(id)
    }

    /// Validate plugin ID format
    fn validate(id: &str) -> PluginResult<()> {
        let parts: Vec<&str> = id.split('.').collect();
        if parts.len() != 2 {
            return Err(PluginError::InvalidPluginId(
                "Plugin ID must be in format 'namespace.name'".to_string(),
            ));
        }

        let namespace = parts[0];
        let name = parts[1];

        // Validate namespace
        if namespace.len() < 2 || namespace.len() > 32 {
            return Err(PluginError::InvalidPluginId(
                "Namespace must be 2-32 characters".to_string(),
            ));
        }
        if !namespace
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(PluginError::InvalidPluginId(
                "Namespace must contain only lowercase letters, digits, and hyphens".to_string(),
            ));
        }

        // Validate name
        if name.len() < 2 || name.len() > 64 {
            return Err(PluginError::InvalidPluginId(
                "Name must be 2-64 characters".to_string(),
            ));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(PluginError::InvalidPluginId(
                "Name must contain only lowercase letters, digits, and hyphens".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the namespace portion of the ID
    pub fn namespace(&self) -> &str {
        self.0.split('.').next().unwrap_or("")
    }

    /// Get the name portion of the ID
    pub fn name(&self) -> &str {
        self.0.split('.').nth(1).unwrap_or("")
    }

    /// Get the full ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PluginId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for PluginId {
    type Err = PluginError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PluginId::new(s)
    }
}

// ============================================================================
// CAPABILITY SET
// ============================================================================

/// Defines what a plugin is allowed to do
///
/// Capabilities follow the principle of least privilege - plugins only get
/// the permissions they explicitly request and are granted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// Can read blockchain state
    pub can_read_state: bool,

    /// Can write to plugin-local state
    pub can_write_state: bool,

    /// Can emit events to the event bus
    pub can_emit_events: bool,

    /// Can call other smart contracts
    pub can_call_contracts: bool,

    /// Can access network (P2P messaging)
    pub can_access_network: bool,

    /// Can use cryptographic primitives
    pub can_access_crypto: bool,

    /// Can access quantum random number generator
    pub can_access_qrng: bool,

    /// Can participate in consensus
    pub can_participate_consensus: bool,

    /// Maximum WASM linear memory pages (64KB each)
    pub max_memory_pages: u32,

    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,

    /// Maximum storage bytes the plugin can use
    pub max_storage_bytes: u64,

    /// Custom capabilities (for extensibility)
    #[serde(default)]
    pub custom_capabilities: HashSet<String>,
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self {
            can_read_state: false,
            can_write_state: false,
            can_emit_events: false,
            can_call_contracts: false,
            can_access_network: false,
            can_access_crypto: false,
            can_access_qrng: false,
            can_participate_consensus: false,
            max_memory_pages: 16, // 1MB default
            max_execution_time_ms: 1000,
            max_storage_bytes: 1024 * 1024, // 1MB default
            custom_capabilities: HashSet::new(),
        }
    }
}

impl CapabilitySet {
    /// Create a minimal capability set (read-only, no side effects)
    pub fn minimal() -> Self {
        Self {
            can_read_state: true,
            max_memory_pages: 4,
            max_execution_time_ms: 100,
            max_storage_bytes: 0,
            ..Default::default()
        }
    }

    /// Create a standard capability set for typical plugins
    pub fn standard() -> Self {
        Self {
            can_read_state: true,
            can_write_state: true,
            can_emit_events: true,
            can_access_crypto: true,
            max_memory_pages: 64, // 4MB
            max_execution_time_ms: 5000,
            max_storage_bytes: 10 * 1024 * 1024, // 10MB
            ..Default::default()
        }
    }

    /// Create a full capability set for trusted system plugins
    pub fn full() -> Self {
        Self {
            can_read_state: true,
            can_write_state: true,
            can_emit_events: true,
            can_call_contracts: true,
            can_access_network: true,
            can_access_crypto: true,
            can_access_qrng: true,
            can_participate_consensus: true,
            max_memory_pages: 256, // 16MB
            max_execution_time_ms: 30000,
            max_storage_bytes: 100 * 1024 * 1024, // 100MB
            custom_capabilities: HashSet::new(),
        }
    }

    /// Check if this capability set satisfies another (is a superset)
    pub fn satisfies(&self, required: &CapabilitySet) -> bool {
        (!required.can_read_state || self.can_read_state)
            && (!required.can_write_state || self.can_write_state)
            && (!required.can_emit_events || self.can_emit_events)
            && (!required.can_call_contracts || self.can_call_contracts)
            && (!required.can_access_network || self.can_access_network)
            && (!required.can_access_crypto || self.can_access_crypto)
            && (!required.can_access_qrng || self.can_access_qrng)
            && (!required.can_participate_consensus || self.can_participate_consensus)
            && self.max_memory_pages >= required.max_memory_pages
            && self.max_execution_time_ms >= required.max_execution_time_ms
            && self.max_storage_bytes >= required.max_storage_bytes
            && required
                .custom_capabilities
                .is_subset(&self.custom_capabilities)
    }

    /// Add a custom capability
    pub fn add_custom_capability(&mut self, capability: impl Into<String>) {
        self.custom_capabilities.insert(capability.into());
    }

    /// Check if a custom capability is present
    pub fn has_custom_capability(&self, capability: &str) -> bool {
        self.custom_capabilities.contains(capability)
    }
}

// ============================================================================
// ENTRY POINTS
// ============================================================================

/// Plugin entry points - hooks into the node's lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntryPoint {
    /// Called when a new block is received
    OnBlockReceived,

    /// Called when a new transaction is received
    OnTransactionReceived,

    /// Called when a validator message is received (for consensus plugins)
    OnValidatorMessage,

    /// Called at the start of each consensus round
    OnConsensusRound,

    /// Called when a new peer connects
    OnPeerConnected,

    /// Called when a peer disconnects
    OnPeerDisconnected,

    /// Called periodically (tick interval configurable)
    OnTick,

    /// Called when the plugin is initialized
    OnInit,

    /// Called when the plugin is being shut down
    OnShutdown,

    /// Called when the plugin is being upgraded
    OnUpgrade,

    /// Custom hook with user-defined name
    CustomHook(String),
}

impl EntryPoint {
    /// Get the WASM export function name for this entry point
    pub fn export_name(&self) -> String {
        match self {
            EntryPoint::OnBlockReceived => "on_block_received".to_string(),
            EntryPoint::OnTransactionReceived => "on_transaction_received".to_string(),
            EntryPoint::OnValidatorMessage => "on_validator_message".to_string(),
            EntryPoint::OnConsensusRound => "on_consensus_round".to_string(),
            EntryPoint::OnPeerConnected => "on_peer_connected".to_string(),
            EntryPoint::OnPeerDisconnected => "on_peer_disconnected".to_string(),
            EntryPoint::OnTick => "on_tick".to_string(),
            EntryPoint::OnInit => "on_init".to_string(),
            EntryPoint::OnShutdown => "on_shutdown".to_string(),
            EntryPoint::OnUpgrade => "on_upgrade".to_string(),
            EntryPoint::CustomHook(name) => format!("hook_{}", name),
        }
    }

    /// Check if this entry point requires specific capabilities
    pub fn required_capabilities(&self) -> CapabilitySet {
        match self {
            EntryPoint::OnBlockReceived | EntryPoint::OnTransactionReceived => CapabilitySet {
                can_read_state: true,
                ..Default::default()
            },
            EntryPoint::OnValidatorMessage | EntryPoint::OnConsensusRound => CapabilitySet {
                can_read_state: true,
                can_participate_consensus: true,
                ..Default::default()
            },
            EntryPoint::OnPeerConnected | EntryPoint::OnPeerDisconnected => CapabilitySet {
                can_access_network: true,
                ..Default::default()
            },
            _ => CapabilitySet::default(),
        }
    }
}

impl fmt::Display for EntryPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryPoint::OnBlockReceived => write!(f, "OnBlockReceived"),
            EntryPoint::OnTransactionReceived => write!(f, "OnTransactionReceived"),
            EntryPoint::OnValidatorMessage => write!(f, "OnValidatorMessage"),
            EntryPoint::OnConsensusRound => write!(f, "OnConsensusRound"),
            EntryPoint::OnPeerConnected => write!(f, "OnPeerConnected"),
            EntryPoint::OnPeerDisconnected => write!(f, "OnPeerDisconnected"),
            EntryPoint::OnTick => write!(f, "OnTick"),
            EntryPoint::OnInit => write!(f, "OnInit"),
            EntryPoint::OnShutdown => write!(f, "OnShutdown"),
            EntryPoint::OnUpgrade => write!(f, "OnUpgrade"),
            EntryPoint::CustomHook(name) => write!(f, "CustomHook({})", name),
        }
    }
}

// ============================================================================
// PLUGIN MANIFEST
// ============================================================================

/// Complete plugin manifest with all metadata and security information
///
/// The manifest must be signed by the author's Ed25519 private key.
/// The signature covers all fields except the signature itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin identifier
    pub id: PluginId,

    /// Human-readable plugin name
    pub name: String,

    /// Semantic version of the plugin
    pub version: semver::Version,

    /// Plugin description
    pub description: String,

    /// Author's Ed25519 public key (32 bytes)
    pub author: [u8; 32],

    /// Author's display name (optional)
    #[serde(default)]
    pub author_name: Option<String>,

    /// SHA3-256 hash of the WASM bytecode
    pub wasm_hash: [u8; 32],

    /// Plugin capabilities (permissions)
    pub capabilities: CapabilitySet,

    /// Entry points the plugin implements
    pub entry_points: Vec<EntryPoint>,

    /// Minimum gas limit required to run this plugin
    pub min_gas_limit: u64,

    /// Plugin dependencies
    #[serde(default)]
    pub dependencies: Vec<PluginDependency>,

    /// Plugin metadata (arbitrary key-value pairs)
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,

    /// Timestamp when the manifest was created (Unix epoch seconds)
    pub created_at: u64,

    /// Ed25519 signature over the manifest (excluding this field)
    #[serde(with = "signature_serde")]
    pub signature: Vec<u8>,
}

/// Serde module for signature serialization
mod signature_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

impl PluginManifest {
    /// Create a new manifest builder
    pub fn builder() -> PluginManifestBuilder {
        PluginManifestBuilder::new()
    }

    /// Compute the hash of all signable fields
    pub fn signable_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Include all fields except signature
        hasher.update(self.id.as_str().as_bytes());
        hasher.update(self.name.as_bytes());
        hasher.update(self.version.to_string().as_bytes());
        hasher.update(self.description.as_bytes());
        hasher.update(&self.author);
        hasher.update(&self.wasm_hash);
        hasher.update(&bincode::serialize(&self.capabilities).unwrap_or_default());
        hasher.update(&bincode::serialize(&self.entry_points).unwrap_or_default());
        hasher.update(&self.min_gas_limit.to_le_bytes());
        hasher.update(&bincode::serialize(&self.dependencies).unwrap_or_default());
        hasher.update(&self.created_at.to_le_bytes());

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Sign the manifest with an Ed25519 private key
    pub fn sign(&mut self, signing_key: &SigningKey) -> PluginResult<()> {
        let hash = self.signable_hash();
        let signature = signing_key.sign(&hash);
        self.signature = signature.to_bytes().to_vec();
        Ok(())
    }

    /// Verify the manifest signature
    pub fn verify_signature(&self) -> PluginResult<bool> {
        let verifying_key = VerifyingKey::from_bytes(&self.author)
            .map_err(|e| PluginError::InvalidAuthorKey(e.to_string()))?;

        let hash = self.signable_hash();

        if self.signature.len() != 64 {
            return Err(PluginError::InvalidSignature(format!(
                "Expected 64 bytes, got {}",
                self.signature.len()
            )));
        }

        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&self.signature);
        let signature = Signature::from_bytes(&sig_bytes);

        match verifying_key.verify(&hash, &signature) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Verify the WASM bytecode hash matches
    pub fn verify_wasm_hash(&self, wasm_bytecode: &[u8]) -> PluginResult<bool> {
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytecode);
        let result = hasher.finalize();

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);

        if hash != self.wasm_hash {
            return Err(PluginError::WasmHashMismatch {
                expected: hex::encode(self.wasm_hash),
                actual: hex::encode(hash),
            });
        }

        Ok(true)
    }

    /// Validate the manifest is well-formed
    pub fn validate(&self) -> PluginResult<()> {
        // Check name
        if self.name.is_empty() || self.name.len() > 128 {
            return Err(PluginError::InvalidManifest(
                "Name must be 1-128 characters".to_string(),
            ));
        }

        // Check entry points
        if self.entry_points.is_empty() {
            return Err(PluginError::InvalidManifest(
                "At least one entry point required".to_string(),
            ));
        }

        // Check each entry point has required capabilities
        for entry_point in &self.entry_points {
            let required = entry_point.required_capabilities();
            if !self.capabilities.satisfies(&required) {
                return Err(PluginError::InsufficientCapabilities {
                    operation: format!("Entry point {} requires capabilities not declared", entry_point),
                });
            }
        }

        // Check gas limit
        if self.min_gas_limit == 0 {
            return Err(PluginError::InvalidManifest(
                "min_gas_limit must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// PLUGIN MANIFEST BUILDER
// ============================================================================

/// Builder for creating plugin manifests
pub struct PluginManifestBuilder {
    id: Option<PluginId>,
    name: Option<String>,
    version: Option<semver::Version>,
    description: String,
    author: Option<[u8; 32]>,
    author_name: Option<String>,
    wasm_hash: Option<[u8; 32]>,
    capabilities: CapabilitySet,
    entry_points: Vec<EntryPoint>,
    min_gas_limit: u64,
    dependencies: Vec<PluginDependency>,
    metadata: std::collections::HashMap<String, String>,
}

impl PluginManifestBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            id: None,
            name: None,
            version: None,
            description: String::new(),
            author: None,
            author_name: None,
            wasm_hash: None,
            capabilities: CapabilitySet::default(),
            entry_points: Vec::new(),
            min_gas_limit: 1000,
            dependencies: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the plugin ID
    pub fn id(mut self, id: PluginId) -> Self {
        self.id = Some(id);
        self
    }

    /// Set the plugin name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the plugin version
    pub fn version(mut self, version: semver::Version) -> Self {
        self.version = Some(version);
        self
    }

    /// Set the plugin description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the author public key
    pub fn author(mut self, author: [u8; 32]) -> Self {
        self.author = Some(author);
        self
    }

    /// Set the author display name
    pub fn author_name(mut self, name: impl Into<String>) -> Self {
        self.author_name = Some(name.into());
        self
    }

    /// Set the WASM hash
    pub fn wasm_hash(mut self, hash: [u8; 32]) -> Self {
        self.wasm_hash = Some(hash);
        self
    }

    /// Compute and set the WASM hash from bytecode
    pub fn wasm_bytecode(mut self, bytecode: &[u8]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(bytecode);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        self.wasm_hash = Some(hash);
        self
    }

    /// Set the capability set
    pub fn capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Add an entry point
    pub fn entry_point(mut self, entry_point: EntryPoint) -> Self {
        self.entry_points.push(entry_point);
        self
    }

    /// Set the minimum gas limit
    pub fn min_gas_limit(mut self, gas: u64) -> Self {
        self.min_gas_limit = gas;
        self
    }

    /// Add a dependency
    pub fn dependency(mut self, dep: PluginDependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the manifest (unsigned)
    pub fn build(self) -> PluginResult<PluginManifest> {
        let manifest = PluginManifest {
            id: self
                .id
                .ok_or_else(|| PluginError::MissingManifestField("id".to_string()))?,
            name: self
                .name
                .ok_or_else(|| PluginError::MissingManifestField("name".to_string()))?,
            version: self
                .version
                .ok_or_else(|| PluginError::MissingManifestField("version".to_string()))?,
            description: self.description,
            author: self
                .author
                .ok_or_else(|| PluginError::MissingManifestField("author".to_string()))?,
            author_name: self.author_name,
            wasm_hash: self
                .wasm_hash
                .ok_or_else(|| PluginError::MissingManifestField("wasm_hash".to_string()))?,
            capabilities: self.capabilities,
            entry_points: self.entry_points,
            min_gas_limit: self.min_gas_limit,
            dependencies: self.dependencies,
            metadata: self.metadata,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            signature: Vec::new(),
        };

        manifest.validate()?;
        Ok(manifest)
    }

    /// Build and sign the manifest
    pub fn build_signed(self, signing_key: &SigningKey) -> PluginResult<PluginManifest> {
        let mut manifest = self.build()?;
        manifest.sign(signing_key)?;
        Ok(manifest)
    }
}

impl Default for PluginManifestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PLUGIN DEPENDENCY
// ============================================================================

/// Plugin dependency specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginDependency {
    /// ID of the required plugin
    pub plugin_id: PluginId,

    /// Semantic version requirement (e.g., ">=1.0.0, <2.0.0")
    pub version_req: String,

    /// Whether this dependency is optional
    #[serde(default)]
    pub optional: bool,
}

impl PluginDependency {
    /// Create a new required dependency
    pub fn required(plugin_id: PluginId, version_req: impl Into<String>) -> Self {
        Self {
            plugin_id,
            version_req: version_req.into(),
            optional: false,
        }
    }

    /// Create a new optional dependency
    pub fn optional(plugin_id: PluginId, version_req: impl Into<String>) -> Self {
        Self {
            plugin_id,
            version_req: version_req.into(),
            optional: true,
        }
    }

    /// Check if a version satisfies this dependency
    pub fn is_satisfied_by(&self, version: &semver::Version) -> PluginResult<bool> {
        let req = semver::VersionReq::parse(&self.version_req)
            .map_err(|e| PluginError::InvalidVersion(e.to_string()))?;
        Ok(req.matches(version))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn test_plugin_id_validation() {
        // Valid IDs
        assert!(PluginId::new("qnk.validator").is_ok());
        assert!(PluginId::new("community.my-plugin").is_ok());
        assert!(PluginId::new("aa.bb").is_ok());

        // Invalid IDs
        assert!(PluginId::new("").is_err());
        assert!(PluginId::new("no-dot").is_err());
        assert!(PluginId::new("QNK.Validator").is_err()); // uppercase
        assert!(PluginId::new("a.b").is_err()); // too short
        assert!(PluginId::new("abc.def.ghi").is_err()); // too many dots
    }

    #[test]
    fn test_capability_set() {
        let minimal = CapabilitySet::minimal();
        let standard = CapabilitySet::standard();
        let full = CapabilitySet::full();

        // Full should satisfy all
        assert!(full.satisfies(&minimal));
        assert!(full.satisfies(&standard));
        assert!(full.satisfies(&full));

        // Standard should satisfy minimal
        assert!(standard.satisfies(&minimal));

        // Minimal should not satisfy standard
        assert!(!minimal.satisfies(&standard));
    }

    #[test]
    fn test_manifest_signing() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let wasm_bytecode = b"(module)";
        let mut hasher = Sha3_256::new();
        hasher.update(wasm_bytecode);
        let result = hasher.finalize();
        let mut wasm_hash = [0u8; 32];
        wasm_hash.copy_from_slice(&result);

        let manifest = PluginManifest::builder()
            .id(PluginId::new("test.plugin").unwrap())
            .name("Test Plugin")
            .version(semver::Version::new(1, 0, 0))
            .author(verifying_key.to_bytes())
            .wasm_hash(wasm_hash)
            .capabilities(CapabilitySet::minimal())
            .entry_point(EntryPoint::OnInit)
            .min_gas_limit(1000)
            .build_signed(&signing_key)
            .unwrap();

        assert!(manifest.verify_signature().unwrap());
    }

    #[test]
    fn test_manifest_validation() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Missing entry points should fail
        let result = PluginManifest::builder()
            .id(PluginId::new("test.plugin").unwrap())
            .name("Test")
            .version(semver::Version::new(1, 0, 0))
            .author(verifying_key.to_bytes())
            .wasm_hash([0u8; 32])
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_entry_point_export_names() {
        assert_eq!(EntryPoint::OnBlockReceived.export_name(), "on_block_received");
        assert_eq!(EntryPoint::OnInit.export_name(), "on_init");
        assert_eq!(
            EntryPoint::CustomHook("my_hook".to_string()).export_name(),
            "hook_my_hook"
        );
    }

    #[test]
    fn test_dependency_satisfaction() {
        let dep = PluginDependency::required(
            PluginId::new("qnk.crypto").unwrap(),
            ">=1.0.0, <2.0.0".to_string(),
        );

        assert!(dep.is_satisfied_by(&semver::Version::new(1, 0, 0)).unwrap());
        assert!(dep.is_satisfied_by(&semver::Version::new(1, 5, 0)).unwrap());
        assert!(!dep.is_satisfied_by(&semver::Version::new(2, 0, 0)).unwrap());
        assert!(!dep.is_satisfied_by(&semver::Version::new(0, 9, 0)).unwrap());
    }
}
