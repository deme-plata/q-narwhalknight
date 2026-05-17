//! Plugin Persistence Module
//!
//! Provides decentralized, consensus-verified storage for plugins and their state.
//! Every node stores and verifies plugin data through P2P distribution and
//! consensus inclusion in DAG-Knight blocks.
//!
//! ## Features
//!
//! - **RocksDB Storage**: Persistent storage with column families for plugins, state, WASM
//! - **Consensus Verification**: State changes require multi-node verification
//! - **P2P Distribution**: Automatic synchronization across all nodes
//! - **Signature Verification**: Ed25519 signatures on all state transitions
//! - **Hash Verification**: SHA3-256 integrity checks on all data
//!
//! ## Usage
//!
//! ```rust,ignore
//! use q_plugin_system::persistence::{PluginPersistentStorage, VerifiedPluginState};
//!
//! // Create persistent storage
//! let storage = PluginPersistentStorage::new(
//!     "./data/plugins",
//!     peer_id,
//!     Some(signing_key),
//! )?;
//!
//! // Store plugin state with automatic signing and hash computation
//! let state = storage.store_plugin_state("my-plugin:1.0.0", &state_data).await?;
//!
//! // Receive and verify state from peer
//! let verified = storage.receive_peer_state("my-plugin:1.0.0", &peer_state, "peer-123").await?;
//!
//! // Check verification status
//! let status = storage.get_verification_status("my-plugin:1.0.0").await?;
//! match status {
//!     PluginVerificationStatus::ConsensusReached => println!("Fully verified!"),
//!     PluginVerificationStatus::Verified => println!("Verified by minimum nodes"),
//!     PluginVerificationStatus::Pending => println!("Awaiting more verifications"),
//!     _ => {}
//! }
//! ```

#[cfg(not(target_os = "windows"))]
mod rocksdb_storage;

#[cfg(not(target_os = "windows"))]
pub use rocksdb_storage::{
    PluginConsensusProof,
    PluginPersistenceError,
    PluginPersistenceStats,
    PluginPersistentStorage,
    PluginVerificationStatus,
    ValidatorSignature,
    VerifiedPluginState,
    CF_PLUGINS,
    CF_PLUGIN_CONSENSUS,
    CF_PLUGIN_STATE,
    CF_PLUGIN_VERIFICATION,
    CF_PLUGIN_WASM,
};
