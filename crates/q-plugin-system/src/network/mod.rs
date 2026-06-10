//! P2P Plugin Distribution Layer for Q-NarwhalKnight
//!
//! This module provides the P2P network layer for distributing plugins across
//! the Q-NarwhalKnight network using libp2p gossipsub.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------+     +-----------------------+
//! |      Node A           |     |      Node B           |
//! +-----------------------+     +-----------------------+
//! |  PluginDistributor    |     |  PluginDistributor    |
//! |  - local_plugins      |<--->|  - local_plugins      |
//! |  - pending_downloads  |     |  - pending_downloads  |
//! +-----------------------+     +-----------------------+
//!         |                              |
//!         v                              v
//! +-----------------------------------------------+
//! |        Gossipsub Topic:                       |
//! |   /qnk/{network_id}/plugins/v1               |
//! +-----------------------------------------------+
//! ```
//!
//! ## Features
//!
//! - **Plugin Announcement**: Broadcast new plugins to the network
//! - **Chunked Transfer**: Large WASM files are split into verifiable chunks
//! - **Registry Sync**: Synchronize plugin registries across nodes
//! - **Cryptographic Verification**: Ed25519 signatures + SHA3-256 hashes
//!
//! ## Security
//!
//! All plugins are verified before installation:
//! - Manifest signature verification (Ed25519)
//! - WASM bytecode hash verification (SHA3-256)
//! - Chunk hash verification during download
//! - Optional trusted author list (governance-controlled)

pub mod distribution;
pub mod protocol;
pub mod verification;

// Re-export main types
pub use distribution::{
    DownloadProgress, DownloadState, PluginDistributor, PluginDistributorConfig,
    PluginDistributorStats,
};
pub use protocol::{
    PluginChunkData, PluginGossipsubMessage, PluginMessage, PluginMessagePriority,
    PLUGIN_PROTOCOL_VERSION, TOPIC_PLUGINS_V1,
};
pub use verification::{
    ManifestVerificationResult, PluginVerificationError, PluginVerifier, TrustedAuthorList,
    WasmVerificationResult,
};

/// Generate the gossipsub topic for plugin distribution
///
/// # Arguments
/// * `network_id` - The network identifier (e.g., "testnet-phase19", "mainnet")
///
/// # Returns
/// The full topic string: `/qnk/{network_id}/plugins/v1`
pub fn plugin_topic(network_id: &str) -> String {
    format!("/qnk/{}/plugins/v1", network_id)
}

/// Default chunk size for plugin distribution (256 KB)
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Maximum plugin size (100 MB)
pub const MAX_PLUGIN_SIZE: usize = 100 * 1024 * 1024;

/// Maximum chunks per plugin (calculated from max size / chunk size)
pub const MAX_CHUNKS_PER_PLUGIN: u32 = (MAX_PLUGIN_SIZE / DEFAULT_CHUNK_SIZE) as u32 + 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_topic_generation() {
        assert_eq!(
            plugin_topic("testnet-phase19"),
            "/qnk/testnet-phase19/plugins/v1"
        );
        assert_eq!(plugin_topic("mainnet"), "/qnk/mainnet/plugins/v1");
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_CHUNK_SIZE, 262144); // 256 KB
        assert_eq!(MAX_PLUGIN_SIZE, 104857600); // 100 MB
        assert!(MAX_CHUNKS_PER_PLUGIN >= 400); // At least 400 chunks for 100MB
    }
}
