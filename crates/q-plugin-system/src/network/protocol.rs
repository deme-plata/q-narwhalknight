//! P2P Protocol Messages for Plugin Distribution
//!
//! Defines the message types used for plugin distribution over libp2p gossipsub.
//! All messages are serialized with bincode for efficient binary transfer.
//!
//! ## Message Flow
//!
//! ```text
//! Publisher                          Subscriber
//!     |                                    |
//!     |--- Announce (manifest, chunks) --->|
//!     |                                    |
//!     |<-- RequestPlugin (id, chunk 0) ---|
//!     |--- PluginChunk (id, 0, data) ---->|
//!     |<-- RequestPlugin (id, chunk 1) ---|
//!     |--- PluginChunk (id, 1, data) ---->|
//!     |          ...                       |
//!     |                                    |
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Protocol version for plugin distribution messages
/// Increment when making breaking changes to message format
pub const PLUGIN_PROTOCOL_VERSION: u32 = 1;

/// Default gossipsub topic suffix for plugin distribution
pub const TOPIC_PLUGINS_V1: &str = "plugins/v1";

/// Plugin manifest - describes a plugin without the bytecode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    /// Unique plugin identifier (e.g., "com.example.my-plugin")
    pub plugin_id: String,
    /// Semantic version (e.g., "1.0.0")
    pub version: String,
    /// Human-readable name
    pub name: String,
    /// Plugin description
    pub description: String,
    /// Author's public key (Ed25519, 32 bytes, hex-encoded)
    pub author_pubkey: String,
    /// SHA3-256 hash of the WASM bytecode (32 bytes, hex-encoded)
    pub wasm_hash: String,
    /// Size of the WASM bytecode in bytes
    pub wasm_size: u64,
    /// Minimum Q-NarwhalKnight version required
    pub min_node_version: String,
    /// Plugin permissions required
    pub permissions: PluginManifestPermissions,
    /// Dependencies on other plugins
    pub dependencies: Vec<PluginDependencySpec>,
    /// Unix timestamp of publication
    pub published_at: u64,
    /// Ed25519 signature of the manifest (excluding this field)
    /// Signs: plugin_id || version || name || author_pubkey || wasm_hash || wasm_size || published_at
    pub signature: Vec<u8>,
}

/// Permissions declared in the plugin manifest
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PluginManifestPermissions {
    /// Access to network operations
    pub network_access: bool,
    /// Access to filesystem (sandboxed)
    pub filesystem_access: bool,
    /// Participate in consensus
    pub consensus_participation: bool,
    /// Modify blockchain state
    pub state_modification: bool,
    /// Process transactions
    pub transaction_processing: bool,
    /// Access to cryptographic operations
    pub crypto_operations: bool,
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginDependencySpec {
    /// Plugin ID of the dependency
    pub plugin_id: String,
    /// Version requirement (semver compatible)
    pub version_requirement: String,
    /// Whether this dependency is optional
    pub optional: bool,
}

/// Unique identifier for a plugin (id + version)
pub type PluginId = String;

/// P2P messages for plugin distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginMessage {
    /// Announce a new plugin to the network
    /// Sent when a node publishes or receives a new plugin
    Announce {
        /// The plugin manifest
        manifest: PluginManifest,
        /// Total number of chunks for this plugin's WASM bytecode
        chunk_count: u32,
    },

    /// Request plugin bytecode chunk from the network
    /// Nodes with the plugin will respond with PluginChunk
    RequestPlugin {
        /// Plugin identifier (plugin_id:version)
        plugin_id: PluginId,
        /// Which chunk to request (0-indexed)
        chunk_index: u32,
    },

    /// Response with plugin bytecode chunk
    /// Sent in response to RequestPlugin
    PluginChunk {
        /// Plugin identifier (plugin_id:version)
        plugin_id: PluginId,
        /// Index of this chunk (0-indexed)
        chunk_index: u32,
        /// Total number of chunks
        total_chunks: u32,
        /// Chunk data (up to CHUNK_SIZE bytes)
        data: Vec<u8>,
        /// SHA3-256 hash of this chunk for verification
        chunk_hash: [u8; 32],
    },

    /// Plugin registry synchronization
    /// Sent periodically or on request to sync known plugins
    RegistrySync {
        /// List of known plugin manifests
        plugins: Vec<PluginManifest>,
    },

    /// Request registry sync from peers
    RequestRegistrySync {
        /// Optional filter: only plugins updated after this timestamp
        since_timestamp: Option<u64>,
        /// Maximum number of plugins to return
        max_count: Option<u32>,
    },

    /// Plugin availability query
    /// Ask if any peer has a specific plugin
    QueryAvailability {
        /// Plugin identifier to query
        plugin_id: PluginId,
    },

    /// Response to availability query
    AvailabilityResponse {
        /// Plugin identifier
        plugin_id: PluginId,
        /// Whether this node has the plugin
        available: bool,
        /// Manifest if available
        manifest: Option<PluginManifest>,
    },
}

/// Message priority for gossipsub routing
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PluginMessagePriority {
    /// Low priority: registry syncs, availability queries
    Low = 0,
    /// Normal priority: plugin requests
    Normal = 1,
    /// High priority: announcements, chunks
    High = 2,
}

impl Default for PluginMessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl PluginMessage {
    /// Get the priority of this message for routing
    pub fn priority(&self) -> PluginMessagePriority {
        match self {
            PluginMessage::Announce { .. } => PluginMessagePriority::High,
            PluginMessage::PluginChunk { .. } => PluginMessagePriority::High,
            PluginMessage::RequestPlugin { .. } => PluginMessagePriority::Normal,
            PluginMessage::RegistrySync { .. } => PluginMessagePriority::Low,
            PluginMessage::RequestRegistrySync { .. } => PluginMessagePriority::Low,
            PluginMessage::QueryAvailability { .. } => PluginMessagePriority::Low,
            PluginMessage::AvailabilityResponse { .. } => PluginMessagePriority::Normal,
        }
    }

    /// Get a human-readable description of the message type
    pub fn message_type(&self) -> &'static str {
        match self {
            PluginMessage::Announce { .. } => "Announce",
            PluginMessage::RequestPlugin { .. } => "RequestPlugin",
            PluginMessage::PluginChunk { .. } => "PluginChunk",
            PluginMessage::RegistrySync { .. } => "RegistrySync",
            PluginMessage::RequestRegistrySync { .. } => "RequestRegistrySync",
            PluginMessage::QueryAvailability { .. } => "QueryAvailability",
            PluginMessage::AvailabilityResponse { .. } => "AvailabilityResponse",
        }
    }
}

/// Gossipsub message envelope for plugin messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginGossipsubMessage {
    /// Protocol version for compatibility
    pub protocol_version: u32,
    /// Unique message identifier
    pub message_id: String,
    /// Unix timestamp of message creation
    pub timestamp: i64,
    /// Sender's peer ID (base58 encoded)
    pub sender_peer_id: String,
    /// The actual plugin message
    pub payload: PluginMessage,
    /// Sequence number for deduplication
    pub sequence_number: u64,
}

impl PluginGossipsubMessage {
    /// Create a new gossipsub message
    pub fn new(sender_peer_id: String, payload: PluginMessage, sequence_number: u64) -> Self {
        Self {
            protocol_version: PLUGIN_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_peer_id,
            payload,
            sequence_number,
        }
    }

    /// Serialize the message for transmission
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize a message from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }

    /// Check if the message timestamp is within acceptable bounds
    /// Rejects messages older than 5 minutes or more than 30s in the future
    pub fn is_timestamp_valid(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        let age = now - self.timestamp;

        // Reject messages older than 5 minutes
        if age > 300 {
            return false;
        }

        // Reject messages more than 30 seconds in the future (clock skew tolerance)
        if age < -30 {
            return false;
        }

        true
    }
}

/// Plugin chunk data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginChunkData {
    /// Plugin identifier
    pub plugin_id: PluginId,
    /// Chunk index
    pub chunk_index: u32,
    /// Total chunks
    pub total_chunks: u32,
    /// Chunk data
    pub data: Vec<u8>,
    /// SHA3-256 hash of this chunk
    pub chunk_hash: [u8; 32],
}

impl PluginChunkData {
    /// Create a new chunk from data
    pub fn new(
        plugin_id: PluginId,
        chunk_index: u32,
        total_chunks: u32,
        data: Vec<u8>,
    ) -> Self {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&data);
        let hash_result = hasher.finalize();
        let mut chunk_hash = [0u8; 32];
        chunk_hash.copy_from_slice(&hash_result);

        Self {
            plugin_id,
            chunk_index,
            total_chunks,
            data,
            chunk_hash,
        }
    }

    /// Verify the chunk hash
    pub fn verify_hash(&self) -> bool {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&self.data);
        let computed_hash = hasher.finalize();

        computed_hash.as_slice() == self.chunk_hash
    }
}

impl PluginManifest {
    /// Create the canonical bytes for signing
    /// Format: plugin_id || version || name || author_pubkey || wasm_hash || wasm_size (LE) || published_at (LE)
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(256);

        // Length-prefixed strings to prevent ambiguity attacks
        fn write_string(buf: &mut Vec<u8>, s: &str) {
            let len = s.len() as u32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        write_string(&mut bytes, &self.plugin_id);
        write_string(&mut bytes, &self.version);
        write_string(&mut bytes, &self.name);
        write_string(&mut bytes, &self.author_pubkey);
        write_string(&mut bytes, &self.wasm_hash);
        bytes.extend_from_slice(&self.wasm_size.to_le_bytes());
        bytes.extend_from_slice(&self.published_at.to_le_bytes());

        bytes
    }

    /// Get the unique identifier for this plugin version
    pub fn unique_id(&self) -> PluginId {
        format!("{}:{}", self.plugin_id, self.version)
    }

    /// Calculate the number of chunks needed for this plugin
    pub fn chunk_count(&self, chunk_size: usize) -> u32 {
        let chunks = (self.wasm_size as usize + chunk_size - 1) / chunk_size;
        chunks as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manifest_signing_bytes() {
        let manifest = PluginManifest {
            plugin_id: "test.plugin".to_string(),
            version: "1.0.0".to_string(),
            name: "Test Plugin".to_string(),
            description: "A test plugin".to_string(),
            author_pubkey: "abc123".to_string(),
            wasm_hash: "deadbeef".to_string(),
            wasm_size: 1024,
            min_node_version: "0.1.0".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 1700000000,
            signature: vec![],
        };

        let bytes = manifest.signing_bytes();
        assert!(!bytes.is_empty());

        // Verify deterministic output
        let bytes2 = manifest.signing_bytes();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn test_plugin_manifest_unique_id() {
        let manifest = PluginManifest {
            plugin_id: "com.example.plugin".to_string(),
            version: "2.1.0".to_string(),
            name: "Example".to_string(),
            description: "".to_string(),
            author_pubkey: "".to_string(),
            wasm_hash: "".to_string(),
            wasm_size: 0,
            min_node_version: "".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 0,
            signature: vec![],
        };

        assert_eq!(manifest.unique_id(), "com.example.plugin:2.1.0");
    }

    #[test]
    fn test_plugin_chunk_data_hash() {
        let chunk = PluginChunkData::new(
            "test:1.0.0".to_string(),
            0,
            1,
            vec![1, 2, 3, 4, 5],
        );

        assert!(chunk.verify_hash());

        // Corrupt the data and verify hash fails
        let mut corrupted = chunk.clone();
        corrupted.data[0] = 255;
        assert!(!corrupted.verify_hash());
    }

    #[test]
    fn test_plugin_message_priority() {
        let announce = PluginMessage::Announce {
            manifest: PluginManifest {
                plugin_id: "test".to_string(),
                version: "1.0.0".to_string(),
                name: "Test".to_string(),
                description: "".to_string(),
                author_pubkey: "".to_string(),
                wasm_hash: "".to_string(),
                wasm_size: 0,
                min_node_version: "".to_string(),
                permissions: PluginManifestPermissions::default(),
                dependencies: vec![],
                published_at: 0,
                signature: vec![],
            },
            chunk_count: 1,
        };

        assert_eq!(announce.priority(), PluginMessagePriority::High);

        let request = PluginMessage::RequestPlugin {
            plugin_id: "test:1.0.0".to_string(),
            chunk_index: 0,
        };

        assert_eq!(request.priority(), PluginMessagePriority::Normal);
    }

    #[test]
    fn test_gossipsub_message_serialization() {
        let msg = PluginGossipsubMessage::new(
            "12D3KooW...".to_string(),
            PluginMessage::QueryAvailability {
                plugin_id: "test:1.0.0".to_string(),
            },
            42,
        );

        let serialized = msg.serialize().unwrap();
        let deserialized = PluginGossipsubMessage::deserialize(&serialized).unwrap();

        assert_eq!(msg.message_id, deserialized.message_id);
        assert_eq!(msg.sequence_number, deserialized.sequence_number);
    }

    #[test]
    fn test_timestamp_validation() {
        let mut msg = PluginGossipsubMessage::new(
            "peer".to_string(),
            PluginMessage::QueryAvailability {
                plugin_id: "test:1.0.0".to_string(),
            },
            1,
        );

        // Current timestamp should be valid
        assert!(msg.is_timestamp_valid());

        // 4 minutes ago should be valid
        msg.timestamp = chrono::Utc::now().timestamp() - 240;
        assert!(msg.is_timestamp_valid());

        // 6 minutes ago should be invalid
        msg.timestamp = chrono::Utc::now().timestamp() - 360;
        assert!(!msg.is_timestamp_valid());

        // 1 minute in the future should be invalid
        msg.timestamp = chrono::Utc::now().timestamp() + 60;
        assert!(!msg.is_timestamp_valid());
    }

    #[test]
    fn test_manifest_chunk_count() {
        let mut manifest = PluginManifest {
            plugin_id: "test".to_string(),
            version: "1.0.0".to_string(),
            name: "Test".to_string(),
            description: "".to_string(),
            author_pubkey: "".to_string(),
            wasm_hash: "".to_string(),
            wasm_size: 0,
            min_node_version: "".to_string(),
            permissions: PluginManifestPermissions::default(),
            dependencies: vec![],
            published_at: 0,
            signature: vec![],
        };

        // 256 KB chunks
        let chunk_size = 256 * 1024;

        // Empty file = 0 chunks
        manifest.wasm_size = 0;
        assert_eq!(manifest.chunk_count(chunk_size), 0);

        // 1 byte = 1 chunk
        manifest.wasm_size = 1;
        assert_eq!(manifest.chunk_count(chunk_size), 1);

        // Exactly 1 chunk size = 1 chunk
        manifest.wasm_size = chunk_size as u64;
        assert_eq!(manifest.chunk_count(chunk_size), 1);

        // 1 byte over = 2 chunks
        manifest.wasm_size = chunk_size as u64 + 1;
        assert_eq!(manifest.chunk_count(chunk_size), 2);

        // 1 MB = 4 chunks
        manifest.wasm_size = 1024 * 1024;
        assert_eq!(manifest.chunk_count(chunk_size), 4);
    }
}
