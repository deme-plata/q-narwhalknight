//! Decentralized Mining Pool Infrastructure
//!
//! v2.3.0-beta: P2P mining pool with CRDT-based PPLNS and gossipsub coordination
//!
//! This module transforms the Q-NarwhalKnight mining pool from a centralized
//! Stratum server to a fully decentralized P2P mining pool.
//!
//! ## Architecture
//!
//! ```text
//!     Pool Node A ◄───► Gossipsub Mesh ◄───► Pool Node B
//!         │                   │                   │
//!     Stratum:3333      /qnk/pool/*          Stratum:3333
//!         │                   │                   │
//!     Miners A,B,C      CRDT State          Miners D,E,F
//! ```
//!
//! ## Key Components
//!
//! - `DistributedShare`: Share with VDF proof for anti-grinding
//! - `DistributedPPLNS`: CRDT-based PPLNS state machine
//! - `PoolTopics`: Gossipsub topics for pool coordination
//! - `PoolNodeDiscovery`: Kademlia-based pool node discovery
//! - `BlockConsensus`: Multi-node block found attestation
//! - `PayoutConsensus`: Threshold signature payout authorization

pub mod share;
pub mod pplns_crdt;
pub mod topics;
pub mod discovery;
pub mod block_consensus;
pub mod payout_consensus;
pub mod coordinator;
pub mod share_ledger;  // v2.3.5-beta: CRDT share ledger with Merkle proofs
pub mod merkle;        // v2.3.5-beta: Merkle tree for share verification

pub use share::{DistributedShare, ShareProof, ShareValidationProof};
pub use pplns_crdt::{DistributedPPLNS, PPLNSStateHash};
pub use topics::PoolTopics;
pub use share_ledger::{ShareLedger, ShareLedgerEntry, ShareLedgerCheckpoint, ShareLedgerMessage};
pub use merkle::{MerkleTree, MerkleProof, MerkleRoot};
pub use discovery::{PoolNodeDiscovery, PoolNodeInfo};
pub use block_consensus::{BlockFoundAnnouncement, NodeAttestation};
pub use payout_consensus::{PayoutBatch, PayoutVote, ThresholdSignature};
pub use coordinator::DistributedPoolCoordinator;

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Peer ID type (libp2p PeerId serializable form)
pub type PeerIdBytes = [u8; 32];

/// Share ID (unique identifier)
pub type ShareId = [u8; 32];

/// Transaction ID
pub type TransactionId = [u8; 32];

/// Pool network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolMessage {
    /// New share announcement
    Share(DistributedShare),

    /// Block found by pool
    BlockFound(BlockFoundAnnouncement),

    /// PPLNS state synchronization
    PPLNSSync(PPLNSSyncMessage),

    /// Payout batch announcement
    Payout(PayoutBatch),

    /// Pool node heartbeat
    Heartbeat(PoolHeartbeat),

    /// Block template distribution
    BlockTemplate(BlockTemplateMessage),

    /// Request PPLNS state from peers
    RequestPPLNSState { from_round: u64 },

    /// Node attestation for block
    Attestation(NodeAttestation),

    /// Payout vote
    PayoutVoteMsg(PayoutVote),
}

/// PPLNS state sync message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPLNSSyncMessage {
    /// Sender's peer ID
    pub sender: PeerIdBytes,

    /// PPLNS state snapshot
    pub state: DistributedPPLNS,

    /// State hash for quick comparison
    pub state_hash: [u8; 32],

    /// Timestamp
    pub timestamp: u64,
}

/// Pool node heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolHeartbeat {
    /// Node peer ID
    pub peer_id: PeerIdBytes,

    /// Stratum port
    pub stratum_port: u16,

    /// Current worker count
    pub worker_count: u32,

    /// Current hashrate (H/s)
    pub hashrate: f64,

    /// Current block height
    pub block_height: u64,

    /// PPLNS state hash
    pub pplns_state_hash: [u8; 32],

    /// Timestamp
    pub timestamp: u64,

    /// Node signature
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

/// Block template message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTemplateMessage {
    /// Template hash
    pub template_hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Previous block hash
    pub prev_hash: [u8; 32],

    /// Merkle root
    pub merkle_root: [u8; 32],

    /// Target difficulty
    pub target: [u8; 32],

    /// Timestamp
    pub timestamp: u64,

    /// Extra nonce space start
    pub extranonce1: Vec<u8>,

    /// Extra nonce 2 size
    pub extranonce2_size: u8,

    /// Coinbase transaction template
    pub coinbase_template: Vec<u8>,

    /// Sender node
    pub sender: PeerIdBytes,

    /// Signature
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

/// Result type for distributed pool operations
pub type DistributedResult<T> = Result<T, DistributedError>;

/// Errors in distributed pool operations
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("Invalid share proof: {0}")]
    InvalidShareProof(String),

    #[error("Share too old: {age_ms}ms (max: {max_ms}ms)")]
    ShareTooOld { age_ms: u64, max_ms: u64 },

    #[error("Duplicate share: {0}")]
    DuplicateShare(String),

    #[error("Invalid signature from node: {0}")]
    InvalidSignature(String),

    #[error("Insufficient attestations: {have}/{need}")]
    InsufficientAttestations { have: usize, need: usize },

    #[error("PPLNS state mismatch: local={local:?}, remote={remote:?}")]
    PPLNSStateMismatch { local: [u8; 32], remote: [u8; 32] },

    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Consensus not reached: {0}")]
    ConsensusNotReached(String),

    #[error("Unknown pool node: {0}")]
    UnknownNode(String),
}
