//! Consensus message types for mempool communication
//!
//! Based on WHAT_HAPPENS_AFTER_CONNECTION.md message specifications

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core consensus message types exchanged between validators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Announce a new transaction to peers
    TransactionAnnounce(TransactionAnnounce),
    /// Request transaction data from a peer
    TransactionRequest(TransactionRequest),
    /// Response with transaction data
    TransactionResponse(TransactionResponse),
    /// Synchronize mempool state with peers
    MempoolSync(MempoolSyncRequest),
    /// Periodic heartbeat message
    Heartbeat(HeartbeatMessage),
}

/// Transaction announcement message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionAnnounce {
    /// Hash of the transaction
    pub tx_hash: [u8; 32],
    /// Transaction size in bytes
    pub size: u32,
    /// Transaction fee (v2.5.0: u128 for extreme precision)
    pub fee: u128,
    /// Timestamp when transaction was received
    pub timestamp: u64,
    /// Optional transaction priority hint
    pub priority: TransactionPriority,
}

/// Transaction request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRequest {
    /// Hashes of requested transactions
    pub tx_hashes: Vec<[u8; 32]>,
    /// Validator requesting the transactions
    pub requestor: String,
    /// Request timestamp
    pub timestamp: u64,
}

/// Transaction response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResponse {
    /// Map of transaction hash to transaction data
    pub transactions: HashMap<[u8; 32], Vec<u8>>,
    /// Validator responding to the request
    pub responder: String,
    /// Response timestamp
    pub timestamp: u64,
}

/// Mempool synchronization request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolSyncRequest {
    /// Type of synchronization requested
    pub sync_type: MempoolSyncType,
    /// Validator requesting sync
    pub requestor: String,
    /// Current mempool state summary
    pub mempool_summary: MempoolSummary,
    /// Request timestamp
    pub timestamp: u64,
}

/// Types of mempool synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MempoolSyncType {
    /// Full mempool state synchronization
    Full,
    /// Incremental sync (transactions after timestamp)
    Incremental { since: u64 },
    /// Missing transactions only
    MissingOnly { known_hashes: Vec<[u8; 32]> },
}

/// Summary of mempool state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolSummary {
    /// Total number of pending transactions
    pub pending_count: u32,
    /// Total mempool size in bytes
    pub total_size: u64,
    /// Hash of mempool state (for consistency checking)
    pub state_hash: [u8; 32],
    /// Highest transaction fee in mempool
    pub max_fee: u64,
    /// Latest transaction timestamp
    pub latest_timestamp: u64,
}

/// Periodic heartbeat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Validator sending heartbeat
    pub validator: String,
    /// Current validator status
    pub status: ValidatorStatus,
    /// Current DAG height
    pub dag_height: u64,
    /// Current mempool size
    pub mempool_size: u32,
    /// Heartbeat timestamp
    pub timestamp: u64,
    /// Network peers known to this validator
    pub known_peers: u32,
}

/// Validator status for heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorStatus {
    /// Validator is active and participating
    Active,
    /// Validator is syncing with network
    Syncing,
    /// Validator is temporarily offline
    Offline,
    /// Validator is in maintenance mode
    Maintenance,
}

/// Transaction priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionPriority {
    /// Low priority (fees < 1000 nanoORB)
    Low,
    /// Normal priority (fees 1000-10000 nanoORB)
    Normal, 
    /// High priority (fees > 10000 nanoORB)
    High,
    /// Critical system transactions
    Critical,
}

impl ConsensusMessage {
    /// Get message type as string for logging
    pub fn message_type(&self) -> &'static str {
        match self {
            ConsensusMessage::TransactionAnnounce(_) => "tx_announce",
            ConsensusMessage::TransactionRequest(_) => "tx_request",
            ConsensusMessage::TransactionResponse(_) => "tx_response", 
            ConsensusMessage::MempoolSync(_) => "mempool_sync",
            ConsensusMessage::Heartbeat(_) => "heartbeat",
        }
    }

    /// Get message priority for network transmission
    pub fn network_priority(&self) -> NetworkPriority {
        match self {
            ConsensusMessage::TransactionAnnounce(tx) => match tx.priority {
                TransactionPriority::Critical => NetworkPriority::Critical,
                TransactionPriority::High => NetworkPriority::High,
                _ => NetworkPriority::Medium,
            },
            ConsensusMessage::TransactionRequest(_) => NetworkPriority::Medium,
            ConsensusMessage::TransactionResponse(_) => NetworkPriority::Medium,
            ConsensusMessage::MempoolSync(_) => NetworkPriority::Low,
            ConsensusMessage::Heartbeat(_) => NetworkPriority::Low,
        }
    }
}

/// Network transmission priority
#[derive(Debug, Clone, Copy)]
pub enum NetworkPriority {
    Critical,
    High, 
    Medium,
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_message_serialization() {
        let tx_announce = TransactionAnnounce {
            tx_hash: [0u8; 32],
            size: 256,
            fee: 5000,
            timestamp: 1693900800,
            priority: TransactionPriority::Normal,
        };

        let message = ConsensusMessage::TransactionAnnounce(tx_announce);
        
        // Test JSON serialization
        let json = serde_json::to_string(&message).unwrap();
        let deserialized: ConsensusMessage = serde_json::from_str(&json).unwrap();
        
        assert_eq!(message.message_type(), "tx_announce");
        matches!(deserialized, ConsensusMessage::TransactionAnnounce(_));
    }

    #[test]
    fn test_network_priority_mapping() {
        let high_fee_tx = TransactionAnnounce {
            tx_hash: [1u8; 32],
            size: 256,
            fee: 15000,
            timestamp: 1693900800,
            priority: TransactionPriority::High,
        };

        let message = ConsensusMessage::TransactionAnnounce(high_fee_tx);
        matches!(message.network_priority(), NetworkPriority::High);
    }
}