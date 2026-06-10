/// Q-Bounty-Protocol: Testnet Bounty Protocol for Q-NarwhalKnight
///
/// A trust-minimized reward system for incentivizing testnet participation.
/// Features:
/// - RocksDB persistent storage for all bounty data
/// - AEGIS-QL post-quantum access control for administrative operations
/// - Weighted scoring system across 5 activity categories
/// - Fraud detection and Sybil resistance
/// - Merkle tree-based mainnet claim proofs
/// - Social media integration for engagement rewards

pub mod types;
pub mod scoring;

#[cfg(not(target_os = "windows"))]
pub mod storage;
#[cfg(target_os = "windows")]
pub mod storage_sled;
#[cfg(target_os = "windows")]
pub use storage_sled as storage;

pub use types::*;
pub use storage::BountyStorage;
pub use scoring::ScoringEngine;

/// Bounty protocol version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default bounty database path
pub const DEFAULT_BOUNTY_DB_PATH: &str = "./data/bounty";
