//! 🎻 Resonance Gossip Protocol
//!
//! This module implements the libp2p gossipsub protocol for distributing
//! resonance states across the network.
//!
//! Philosophy: Like sound waves propagating through air, resonance states
//! propagate through the network. Each validator vibrates at its own frequency,
//! and gossip allows these vibrations to reach all nodes, creating a
//! distributed symphony.

use crate::{
    string_state::StringState,
    vertex::ResonanceVertex,
    ResonanceError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info, warn};
use bincode;

/// 🎻 Protocol ID for resonance gossip
pub const RESONANCE_PROTOCOL: &str = "/qnk/resonance/1.0.0";

/// 🎻 Resonance gossip message types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResonanceMessage {
    /// 🎻 Broadcast string state to network (frequency announcement)
    StringStateAnnouncement {
        round: u64,
        vertex_hash: [u8; 32],
        string_state: StringState,
        validator: Vec<u8>,
        timestamp: u64,
    },

    /// 🎻 Request resonance state from peers (ask others to share their vibration)
    StateRequest {
        round: u64,
        requesting_node: Vec<u8>,
    },

    /// 🎻 Respond with resonance state (share your vibration)
    StateResponse {
        round: u64,
        vertices: Vec<ResonanceVertex>,
        responding_node: Vec<u8>,
    },

    /// 🎻 Consensus achieved notification (the symphony has converged)
    ConsensusAchieved {
        round: u64,
        committed_hashes: Vec<[u8; 32]>,
        final_energy: f64,
        spectral_gap: f64,
        node: Vec<u8>,
    },

    /// 🎻 Byzantine node detected warning (dissonance alert)
    ByzantineAlert {
        round: u64,
        suspected_node: Vec<u8>,
        detector_node: Vec<u8>,
        spectral_coefficient: f64,
        timestamp: u64,
    },
}

/// 🎻 Resonance state tracker for gossip synchronization
#[derive(Clone, Debug)]
pub struct ResonanceStateTracker {
    /// String states by round and validator
    states_by_round: Arc<RwLock<HashMap<u64, HashMap<Vec<u8>, StringState>>>>,

    /// Vertices announced for each round
    vertices_by_round: Arc<RwLock<HashMap<u64, Vec<ResonanceVertex>>>>,

    /// Consensus status by round
    consensus_by_round: Arc<RwLock<HashMap<u64, ConsensusInfo>>>,

    /// Byzantine alerts
    byzantine_alerts: Arc<RwLock<Vec<ByzantineAlertInfo>>>,

    /// Our node ID
    node_id: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ConsensusInfo {
    pub committed_hashes: Vec<[u8; 32]>,
    pub final_energy: f64,
    pub spectral_gap: f64,
    pub achieved_at: u64,
}

#[derive(Clone, Debug)]
pub struct ByzantineAlertInfo {
    pub round: u64,
    pub suspected_node: Vec<u8>,
    pub detector_node: Vec<u8>,
    pub spectral_coefficient: f64,
    pub timestamp: u64,
}

impl ResonanceStateTracker {
    /// 🎻 Create new resonance state tracker
    pub fn new(node_id: Vec<u8>) -> Self {
        info!("🎻 Initializing Resonance State Tracker for node {:?}", node_id);

        Self {
            states_by_round: Arc::new(RwLock::new(HashMap::new())),
            vertices_by_round: Arc::new(RwLock::new(HashMap::new())),
            consensus_by_round: Arc::new(RwLock::new(HashMap::new())),
            byzantine_alerts: Arc::new(RwLock::new(Vec::new())),
            node_id,
        }
    }

    /// 🎻 Record string state announcement
    pub fn record_string_state(
        &self,
        round: u64,
        validator: Vec<u8>,
        string_state: StringState,
    ) {
        let mut states = self.states_by_round.write();
        states
            .entry(round)
            .or_insert_with(HashMap::new)
            .insert(validator.clone(), string_state.clone());

        debug!(
            "🎻 Recorded string state: round={}, validator={:?}, freq={:.4}",
            round,
            validator,
            string_state.frequency
        );
    }

    /// 🎻 Record vertex announcement
    pub fn record_vertex(&self, round: u64, vertex: ResonanceVertex) {
        let mut vertices = self.vertices_by_round.write();
        vertices
            .entry(round)
            .or_insert_with(Vec::new)
            .push(vertex.clone());

        debug!(
            "🎻 Recorded vertex: round={}, hash={:?}",
            round,
            vertex.hash
        );
    }

    /// 🎻 Record consensus achievement
    pub fn record_consensus(
        &self,
        round: u64,
        committed_hashes: Vec<[u8; 32]>,
        final_energy: f64,
        spectral_gap: f64,
    ) {
        let mut consensus = self.consensus_by_round.write();
        consensus.insert(
            round,
            ConsensusInfo {
                committed_hashes,
                final_energy,
                spectral_gap,
                achieved_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            },
        );

        info!(
            "🎻 Consensus achieved: round={}, energy={:.4}, spectral_gap={:.4}",
            round, final_energy, spectral_gap
        );
    }

    /// 🎻 Record Byzantine alert
    pub fn record_byzantine_alert(
        &self,
        round: u64,
        suspected_node: Vec<u8>,
        detector_node: Vec<u8>,
        spectral_coefficient: f64,
    ) {
        let mut alerts = self.byzantine_alerts.write();
        alerts.push(ByzantineAlertInfo {
            round,
            suspected_node: suspected_node.clone(),
            detector_node: detector_node.clone(),
            spectral_coefficient,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        });

        warn!(
            "🎻 Byzantine alert: round={}, suspected={:?}, coefficient={:.4}",
            round, suspected_node, spectral_coefficient
        );
    }

    /// 🎻 Get string states for a round
    pub fn get_states_for_round(&self, round: u64) -> HashMap<Vec<u8>, StringState> {
        self.states_by_round
            .read()
            .get(&round)
            .cloned()
            .unwrap_or_default()
    }

    /// 🎻 Get vertices for a round
    pub fn get_vertices_for_round(&self, round: u64) -> Vec<ResonanceVertex> {
        self.vertices_by_round
            .read()
            .get(&round)
            .cloned()
            .unwrap_or_default()
    }

    /// 🎻 Check if consensus achieved for round
    pub fn has_consensus(&self, round: u64) -> bool {
        self.consensus_by_round.read().contains_key(&round)
    }

    /// 🎻 Get consensus info for round
    pub fn get_consensus(&self, round: u64) -> Option<ConsensusInfo> {
        self.consensus_by_round.read().get(&round).cloned()
    }

    /// 🎻 Get Byzantine alerts for round
    pub fn get_byzantine_alerts(&self, round: u64) -> Vec<ByzantineAlertInfo> {
        self.byzantine_alerts
            .read()
            .iter()
            .filter(|alert| alert.round == round)
            .cloned()
            .collect()
    }

    /// 🎻 Get all Byzantine alerts
    pub fn get_all_byzantine_alerts(&self) -> Vec<ByzantineAlertInfo> {
        self.byzantine_alerts.read().clone()
    }

    /// 🎻 Clean up old rounds (garbage collection)
    pub fn cleanup_old_rounds(&self, keep_rounds: u64) {
        let current_round = self.get_latest_round();
        if current_round < keep_rounds {
            return;
        }

        let cutoff = current_round - keep_rounds;

        // Clean states
        self.states_by_round
            .write()
            .retain(|&round, _| round > cutoff);

        // Clean vertices
        self.vertices_by_round
            .write()
            .retain(|&round, _| round > cutoff);

        // Clean consensus info
        self.consensus_by_round
            .write()
            .retain(|&round, _| round > cutoff);

        // Clean old Byzantine alerts (keep last 1000)
        let mut alerts = self.byzantine_alerts.write();
        if alerts.len() > 1000 {
            let drain_count = alerts.len() - 1000;
            alerts.drain(0..drain_count);
        }

        debug!("🎻 Cleaned up rounds before {}", cutoff);
    }

    /// 🎻 Get latest round number
    fn get_latest_round(&self) -> u64 {
        self.states_by_round
            .read()
            .keys()
            .max()
            .copied()
            .unwrap_or(0)
    }
}

/// 🎻 Serialize resonance message for gossip
pub fn serialize_resonance_message(msg: &ResonanceMessage) -> Result<Vec<u8>> {
    bincode::serialize(msg)
        .map_err(|e| ResonanceError::InvalidState(format!("Serialization failed: {}", e)))
}

/// 🎻 Deserialize resonance message from gossip
pub fn deserialize_resonance_message(data: &[u8]) -> Result<ResonanceMessage> {
    bincode::deserialize(data)
        .map_err(|e| ResonanceError::InvalidState(format!("Deserialization failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_tracker_creation() {
        let tracker = ResonanceStateTracker::new(vec![1, 2, 3]);
        assert_eq!(tracker.node_id, vec![1, 2, 3]);
    }

    #[test]
    fn test_record_and_retrieve_state() {
        let tracker = ResonanceStateTracker::new(vec![1, 2, 3]);
        let state = StringState::new(
            100.0,
            1.0,
            vec![0.0, 0.0],
            [1u8; 32],
            1000,
        );

        tracker.record_string_state(1, vec![4, 5, 6], state.clone());

        let states = tracker.get_states_for_round(1);
        assert_eq!(states.len(), 1);
        assert!(states.contains_key(&vec![4, 5, 6]));
    }

    #[test]
    fn test_consensus_tracking() {
        let tracker = ResonanceStateTracker::new(vec![1, 2, 3]);

        assert!(!tracker.has_consensus(1));

        tracker.record_consensus(1, vec![[1u8; 32], [2u8; 32]], 10.5, 2.3);

        assert!(tracker.has_consensus(1));

        let consensus = tracker.get_consensus(1).unwrap();
        assert_eq!(consensus.final_energy, 10.5);
        assert_eq!(consensus.spectral_gap, 2.3);
    }

    #[test]
    fn test_message_serialization() {
        let msg = ResonanceMessage::ConsensusAchieved {
            round: 10,
            committed_hashes: vec![[1u8; 32]],
            final_energy: 42.0,
            spectral_gap: 3.5,
            node: vec![1, 2, 3],
        };

        let serialized = serialize_resonance_message(&msg).unwrap();
        let deserialized = deserialize_resonance_message(&serialized).unwrap();

        match deserialized {
            ResonanceMessage::ConsensusAchieved { round, .. } => {
                assert_eq!(round, 10);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_cleanup_old_rounds() {
        let tracker = ResonanceStateTracker::new(vec![1, 2, 3]);

        // Add states for rounds 1-10
        for round in 1..=10 {
            let state = StringState::new(
                100.0,
                round as f64,
                vec![0.0, 0.0],
                [round as u8; 32],
                1000,
            );
            tracker.record_string_state(round, vec![4, 5, 6], state);
        }

        // Keep only last 5 rounds
        tracker.cleanup_old_rounds(5);

        // Rounds 1-5 should be cleaned
        assert!(tracker.get_states_for_round(3).is_empty());
        assert!(tracker.get_states_for_round(6).len() > 0);
    }
}
