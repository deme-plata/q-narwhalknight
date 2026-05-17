//! Distributed Quantum Neural Oracle (QNO) P2P Protocol
//!
//! This module provides P2P gossipsub integration for QNO prediction staking.
//! It ensures all nodes validate and synchronize stake operations across the network.

use libp2p::gossipsub::{IdentTopic, Topic};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

/// Gossipsub topics for QNO operations
pub const TOPIC_QNO_STAKES: &str = "qnk/qno/stakes/v1";
pub const TOPIC_QNO_CLAIMS: &str = "qnk/qno/claims/v1";
pub const TOPIC_QNO_SYNC: &str = "qnk/qno/sync/v1";
pub const TOPIC_QNO_RESOLUTION: &str = "qnk/qno/resolution/v1";  // v1.4.3: Resolution outcomes
pub const TOPIC_QNO_ORACLE: &str = "qnk/qno/oracle/v1";         // v1.4.3: Oracle data feeds

/// QNO-specific Gossipsub topics manager
pub struct QnoTopics {
    pub stakes: IdentTopic,
    pub claims: IdentTopic,
    pub sync: IdentTopic,
    pub resolution: IdentTopic,  // v1.4.3: Resolution outcomes
    pub oracle: IdentTopic,      // v1.4.3: Oracle data feeds
}

impl QnoTopics {
    pub fn new() -> Self {
        info!("🔮 Initializing QNO Gossipsub topics for P2P stake validation");

        Self {
            stakes: IdentTopic::new(TOPIC_QNO_STAKES),
            claims: IdentTopic::new(TOPIC_QNO_CLAIMS),
            sync: IdentTopic::new(TOPIC_QNO_SYNC),
            resolution: IdentTopic::new(TOPIC_QNO_RESOLUTION),
            oracle: IdentTopic::new(TOPIC_QNO_ORACLE),
        }
    }

    /// Get all QNO topics for subscription
    pub fn all_topics(&self) -> Vec<IdentTopic> {
        vec![
            self.stakes.clone(),
            self.claims.clone(),
            self.sync.clone(),
            self.resolution.clone(),
            self.oracle.clone(),
        ]
    }

    /// Check if a topic hash matches any QNO topic
    pub fn is_qno_topic(&self, topic: &libp2p::gossipsub::TopicHash) -> bool {
        let topic_str = topic.as_str();
        topic_str.starts_with("qnk/qno/")
    }
}

impl Default for QnoTopics {
    fn default() -> Self {
        Self::new()
    }
}

/// QNO P2P message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QnoMessageType {
    /// New stake operation
    Stake,
    /// Unstake operation (early withdrawal)
    Unstake,
    /// Claim rewards
    Claim,
    /// Sync request (ask peers for current state)
    SyncRequest,
    /// Sync response (share current state)
    SyncResponse,
    /// Oracle outcome announcement (v1.4.3)
    OracleOutcome,
    /// Prediction resolution result (v1.4.3)
    Resolution,
    /// Slashing event notification (v1.4.3)
    Slashing,
}

/// QNO message envelope for Gossipsub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnoGossipMessage {
    /// Protocol version (increment on breaking changes)
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u32,

    /// Unique message ID for deduplication
    pub message_id: String,

    /// Unix timestamp (seconds)
    pub timestamp: u64,

    /// Sender node ID
    pub sender_node_id: String,

    /// Sender peer ID (libp2p)
    pub sender_peer_id: String,

    /// Message type
    pub message_type: QnoMessageType,

    /// Serialized payload (JSON)
    pub payload: Vec<u8>,

    /// Ed25519 signature of the payload
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub signature: Option<Vec<u8>>,

    /// Sender's public key for verification
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub sender_public_key: Option<Vec<u8>>,
}

fn default_protocol_version() -> u32 {
    1
}

/// Current QNO protocol version
pub const CURRENT_QNO_PROTOCOL_VERSION: u32 = 1;

impl QnoGossipMessage {
    /// Create a new stake message
    pub fn new_stake(
        sender_node_id: String,
        sender_peer_id: String,
        stake_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::Stake,
            payload: stake_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Create a new unstake message
    pub fn new_unstake(
        sender_node_id: String,
        sender_peer_id: String,
        unstake_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::Unstake,
            payload: unstake_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Create a new claim message
    pub fn new_claim(
        sender_node_id: String,
        sender_peer_id: String,
        claim_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::Claim,
            payload: claim_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Create a new oracle outcome message (v1.4.3)
    pub fn new_oracle_outcome(
        sender_node_id: String,
        sender_peer_id: String,
        outcome_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::OracleOutcome,
            payload: outcome_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Create a new resolution result message (v1.4.3)
    pub fn new_resolution(
        sender_node_id: String,
        sender_peer_id: String,
        resolution_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::Resolution,
            payload: resolution_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Create a new slashing notification message (v1.4.3)
    pub fn new_slashing(
        sender_node_id: String,
        sender_peer_id: String,
        slashing_data: &[u8],
        signature: Option<Vec<u8>>,
        public_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            protocol_version: CURRENT_QNO_PROTOCOL_VERSION,
            message_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_node_id,
            sender_peer_id,
            message_type: QnoMessageType::Slashing,
            payload: slashing_data.to_vec(),
            signature,
            sender_public_key: public_key,
        }
    }

    /// Validate message timestamp (within 5 minutes)
    pub fn is_valid_timestamp(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.abs_diff(self.timestamp) < 300
    }

    /// Validate protocol version compatibility
    pub fn is_compatible_version(&self) -> bool {
        self.protocol_version <= CURRENT_QNO_PROTOCOL_VERSION
    }

    /// Serialize to bytes for gossipsub
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

/// QNO sync request/response for initial state synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnoSyncState {
    /// Node ID requesting/providing sync
    pub node_id: String,
    /// All active staking positions (serialized)
    pub positions: Vec<Vec<u8>>,
    /// All prediction domains (serialized)
    pub domains: Vec<Vec<u8>>,
    /// Global staking stats (serialized)
    pub stats: Vec<u8>,
    /// Timestamp of this snapshot
    pub snapshot_timestamp: u64,
}

/// Message deduplication cache
/// Uses LRU cache to prevent processing duplicate messages
pub struct QnoDeduplicationCache {
    seen_messages: std::collections::HashSet<String>,
    max_size: usize,
    // In production, use an LRU cache with TTL
}

impl QnoDeduplicationCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            seen_messages: std::collections::HashSet::new(),
            max_size,
        }
    }

    /// Check if message was already seen, and mark it as seen if not
    pub fn check_and_insert(&mut self, message_id: &str) -> bool {
        if self.seen_messages.contains(message_id) {
            return true; // Already seen
        }

        // Evict oldest if at capacity (simple implementation)
        if self.seen_messages.len() >= self.max_size {
            // In production, use proper LRU eviction
            self.seen_messages.clear();
        }

        self.seen_messages.insert(message_id.to_string());
        false // Not seen before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qno_topics_creation() {
        let topics = QnoTopics::new();
        assert_eq!(topics.stakes.hash().as_str(), TOPIC_QNO_STAKES);
        assert_eq!(topics.claims.hash().as_str(), TOPIC_QNO_CLAIMS);
        assert_eq!(topics.sync.hash().as_str(), TOPIC_QNO_SYNC);
    }

    #[test]
    fn test_qno_message_serialization() {
        let msg = QnoGossipMessage::new_stake(
            "node1".to_string(),
            "12D3KooWTest".to_string(),
            b"test stake data",
            None,
            None,
        );

        let bytes = msg.to_bytes().unwrap();
        let decoded = QnoGossipMessage::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.sender_node_id, "node1");
        assert_eq!(decoded.payload, b"test stake data");
    }

    #[test]
    fn test_timestamp_validation() {
        let msg = QnoGossipMessage::new_stake(
            "node1".to_string(),
            "peer1".to_string(),
            b"data",
            None,
            None,
        );
        assert!(msg.is_valid_timestamp());
    }

    #[test]
    fn test_deduplication_cache() {
        let mut cache = QnoDeduplicationCache::new(100);

        // First time - not seen
        assert!(!cache.check_and_insert("msg1"));

        // Second time - already seen
        assert!(cache.check_and_insert("msg1"));

        // New message - not seen
        assert!(!cache.check_and_insert("msg2"));
    }
}
