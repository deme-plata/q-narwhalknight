//! Gossipsub Topics for Pool Coordination
//!
//! Defines the P2P messaging topics used by the decentralized mining pool.

use serde::{Deserialize, Serialize};

/// Pool-specific gossipsub topics
#[derive(Debug, Clone)]
pub struct PoolTopics {
    /// Network ID (e.g., "mainnet", "testnet-phase16")
    pub network_id: String,

    /// Share announcements (high volume)
    /// Topic: /qnk/{network}/pool/shares
    pub shares: String,

    /// Block found notifications (rare, high priority)
    /// Topic: /qnk/{network}/pool/blocks-found
    pub blocks_found: String,

    /// PPLNS state synchronization (periodic)
    /// Topic: /qnk/{network}/pool/pplns-state
    pub pplns_state: String,

    /// Payout batch announcements
    /// Topic: /qnk/{network}/pool/payouts
    pub payouts: String,

    /// Pool node heartbeats
    /// Topic: /qnk/{network}/pool/heartbeat
    pub heartbeat: String,

    /// Block template distribution
    /// Topic: /qnk/{network}/pool/templates
    pub block_templates: String,

    /// Node attestations for blocks
    /// Topic: /qnk/{network}/pool/attestations
    pub attestations: String,

    /// Payout votes
    /// Topic: /qnk/{network}/pool/payout-votes
    pub payout_votes: String,

    // ========== v2.3.5-beta: Share Ledger Topics ==========

    /// Share ledger entries (share submissions for audit/proof)
    /// Topic: /qnk/{network}/pool/share-ledger
    pub share_ledger: String,

    /// Share ledger checkpoints (Merkle roots with consensus)
    /// Topic: /qnk/{network}/pool/share-ledger-checkpoints
    pub share_ledger_checkpoints: String,

    /// Share ledger sync (state synchronization requests)
    /// Topic: /qnk/{network}/pool/share-ledger-sync
    pub share_ledger_sync: String,
}

impl PoolTopics {
    /// Create new pool topics for a network
    pub fn new(network_id: &str) -> Self {
        Self {
            network_id: network_id.to_string(),
            shares: format!("/qnk/{}/pool/shares", network_id),
            blocks_found: format!("/qnk/{}/pool/blocks-found", network_id),
            pplns_state: format!("/qnk/{}/pool/pplns-state", network_id),
            payouts: format!("/qnk/{}/pool/payouts", network_id),
            heartbeat: format!("/qnk/{}/pool/heartbeat", network_id),
            block_templates: format!("/qnk/{}/pool/templates", network_id),
            attestations: format!("/qnk/{}/pool/attestations", network_id),
            payout_votes: format!("/qnk/{}/pool/payout-votes", network_id),
            // v2.3.5-beta: Share Ledger Topics
            share_ledger: format!("/qnk/{}/pool/share-ledger", network_id),
            share_ledger_checkpoints: format!("/qnk/{}/pool/share-ledger-checkpoints", network_id),
            share_ledger_sync: format!("/qnk/{}/pool/share-ledger-sync", network_id),
        }
    }

    /// Get all topics as a list
    pub fn all_topics(&self) -> Vec<&str> {
        vec![
            &self.shares,
            &self.blocks_found,
            &self.pplns_state,
            &self.payouts,
            &self.heartbeat,
            &self.block_templates,
            &self.attestations,
            &self.payout_votes,
            // v2.3.5-beta: Share Ledger Topics
            &self.share_ledger,
            &self.share_ledger_checkpoints,
            &self.share_ledger_sync,
        ]
    }

    /// Get high-priority topics (critical for consensus)
    pub fn high_priority_topics(&self) -> Vec<&str> {
        vec![&self.blocks_found, &self.payouts, &self.attestations]
    }

    /// Get high-volume topics (need special handling)
    pub fn high_volume_topics(&self) -> Vec<&str> {
        vec![&self.shares]
    }
}

/// Topic configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicParams {
    /// Desired mesh size
    pub mesh_size: usize,

    /// Minimum mesh size
    pub mesh_low: usize,

    /// Maximum mesh size
    pub mesh_high: usize,

    /// Gossip factor (0.0 - 1.0)
    pub gossip_factor: f64,

    /// History length for duplicate detection
    pub history_length: usize,

    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,

    /// Max transmit size in bytes
    pub max_transmit_size: usize,
}

impl TopicParams {
    /// Parameters for high-volume topics (shares)
    pub fn high_volume() -> Self {
        Self {
            mesh_size: 8,
            mesh_low: 4,
            mesh_high: 12,
            gossip_factor: 0.25, // Lower gossip for high volume
            history_length: 5,
            heartbeat_interval_ms: 1000,
            max_transmit_size: 1024, // 1KB per share
        }
    }

    /// Parameters for critical topics (blocks found)
    pub fn critical() -> Self {
        Self {
            mesh_size: 12,
            mesh_low: 8,
            mesh_high: 16,
            gossip_factor: 1.0, // Maximum redundancy
            history_length: 10,
            heartbeat_interval_ms: 500,
            max_transmit_size: 65536, // 64KB for block data
        }
    }

    /// Parameters for state sync topics
    pub fn state_sync() -> Self {
        Self {
            mesh_size: 6,
            mesh_low: 3,
            mesh_high: 10,
            gossip_factor: 0.5,
            history_length: 3,
            heartbeat_interval_ms: 2000,
            max_transmit_size: 1048576, // 1MB for state snapshots
        }
    }

    /// Parameters for heartbeat topic
    pub fn heartbeat() -> Self {
        Self {
            mesh_size: 4,
            mesh_low: 2,
            mesh_high: 8,
            gossip_factor: 0.5,
            history_length: 2,
            heartbeat_interval_ms: 5000,
            max_transmit_size: 512,
        }
    }
}

/// Get recommended parameters for a topic
pub fn get_topic_params(topic: &str) -> TopicParams {
    if topic.contains("/shares") {
        TopicParams::high_volume()
    } else if topic.contains("/blocks-found") || topic.contains("/payouts") {
        TopicParams::critical()
    } else if topic.contains("/pplns-state") {
        TopicParams::state_sync()
    } else if topic.contains("/heartbeat") {
        TopicParams::heartbeat()
    } else {
        TopicParams::state_sync() // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_creation() {
        let topics = PoolTopics::new("testnet-phase16");

        assert_eq!(topics.shares, "/qnk/testnet-phase16/pool/shares");
        assert_eq!(
            topics.blocks_found,
            "/qnk/testnet-phase16/pool/blocks-found"
        );
    }

    #[test]
    fn test_all_topics() {
        let topics = PoolTopics::new("mainnet-genesis");
        let all = topics.all_topics();

        // v2.3.5-beta: Now 11 topics (8 original + 3 share ledger topics)
        assert_eq!(all.len(), 11);
        assert!(all.iter().all(|t| t.contains("/qnk/mainnet-genesis/pool/")));
    }
}
