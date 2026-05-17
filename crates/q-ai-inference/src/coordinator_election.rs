//! Coordinator Election for Distributed AI Inference
//!
//! This module implements a democratic coordinator election system based on:
//! - Node capability scores (hardware performance)
//! - Node uptime and reliability
//! - Network latency measurements
//! - Current load and availability
//!
//! The coordinator is responsible for:
//! - Assigning layers to participating nodes
//! - Orchestrating inference requests
//! - Monitoring node health and performance
//! - Reassigning layers on node failure

use crate::types::DeviceCapability;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Election state for coordinator selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionCandidate {
    pub node_id: String,
    pub peer_id: String,
    pub capability: DeviceCapability,
    pub uptime_secs: u64,
    pub inference_count: u64,
    pub average_latency_ms: u64,
    pub last_heartbeat: i64,
    pub election_score: u64,
    /// v6.0.0: Staked QUG amount (24-decimal). Higher stake = higher priority.
    #[serde(default)]
    pub stake_amount: u128,
}

impl ElectionCandidate {
    /// Calculate election score based on multiple factors
    /// v6.0.0: Added stake_weight for economic security
    pub fn calculate_election_score(&mut self) {
        let capability_score = self.capability.score();
        let uptime_score = (self.uptime_secs / 60).min(1000); // Max 1000 points for uptime
        let latency_penalty = if self.average_latency_ms > 0 {
            1000 / (self.average_latency_ms + 1)
        } else {
            1000
        };
        let reliability_score = (self.inference_count / 10).min(500); // Max 500 points for reliability

        // v6.0.0: Stake-weighted scoring
        // log2(stake / 1000_QUG) * 10, capped at 30 points
        // Bronze (1K) = 0, Silver (10K) = 10, Gold (100K) = 20, Diamond (1M) = 30
        let stake_qug = self.stake_amount / 10u128.pow(24);
        let stake_weight = if stake_qug >= 1000 {
            let log_val = (stake_qug / 1000) as f64;
            (log_val.log2() * 10.0).min(30.0) as u64
        } else {
            0
        };

        self.election_score = capability_score + uptime_score + latency_penalty + reliability_score + stake_weight;
        debug!(
            "📊 Election score for {}: {} (cap={}, uptime={}, latency={}, reliability={}, stake={})",
            self.node_id,
            self.election_score,
            capability_score,
            uptime_score,
            latency_penalty,
            reliability_score,
            stake_weight
        );
    }
}

/// Coordinator election manager
pub struct CoordinatorElection {
    /// This node's ID
    node_id: String,

    /// This node's peer ID
    peer_id: String,

    /// This node's capability
    capability: DeviceCapability,

    /// Election candidates (node_id -> candidate)
    candidates: Arc<RwLock<HashMap<String, ElectionCandidate>>>,

    /// Current coordinator
    current_coordinator: Arc<RwLock<Option<String>>>,

    /// Election start time
    election_start: Arc<RwLock<Option<Instant>>>,

    /// Election timeout (default 30 seconds)
    election_timeout: Duration,

    /// Node start time for uptime calculation
    node_start_time: Instant,

    /// Inference statistics
    inference_count: Arc<RwLock<u64>>,
    average_latency_ms: Arc<RwLock<u64>>,
}

impl CoordinatorElection {
    /// Create a new coordinator election manager
    pub fn new(node_id: String, peer_id: String, capability: DeviceCapability) -> Self {
        info!("🗳️ Initializing coordinator election for node: {}", node_id);

        Self {
            node_id,
            peer_id,
            capability,
            candidates: Arc::new(RwLock::new(HashMap::new())),
            current_coordinator: Arc::new(RwLock::new(None)),
            election_start: Arc::new(RwLock::new(None)),
            election_timeout: Duration::from_secs(30),
            node_start_time: Instant::now(),
            inference_count: Arc::new(RwLock::new(0)),
            average_latency_ms: Arc::new(RwLock::new(0)),
        }
    }

    /// Start a new election
    pub async fn start_election(&self) -> Result<()> {
        info!("🗳️ Starting coordinator election");

        let mut election_start = self.election_start.write().await;
        *election_start = Some(Instant::now());

        // Add self as candidate
        let mut candidates = self.candidates.write().await;
        let uptime_secs = self.node_start_time.elapsed().as_secs();
        let inference_count = *self.inference_count.read().await;
        let average_latency_ms = *self.average_latency_ms.read().await;

        let mut self_candidate = ElectionCandidate {
            node_id: self.node_id.clone(),
            peer_id: self.peer_id.clone(),
            capability: self.capability.clone(),
            uptime_secs,
            inference_count,
            average_latency_ms,
            last_heartbeat: chrono::Utc::now().timestamp(),
            election_score: 0,
            stake_amount: 0, // v6.0.0: Updated by worker registry when staking
        };

        self_candidate.calculate_election_score();
        candidates.insert(self.node_id.clone(), self_candidate);

        info!("✅ Election started with {} initial candidates", candidates.len());
        Ok(())
    }

    /// Add a candidate to the election
    pub async fn add_candidate(&self, candidate: ElectionCandidate) -> Result<()> {
        debug!("📥 Adding election candidate: {}", candidate.node_id);

        let mut candidates = self.candidates.write().await;
        candidates.insert(candidate.node_id.clone(), candidate);

        Ok(())
    }

    /// Check if election timeout has been reached
    pub async fn is_election_complete(&self) -> bool {
        let election_start = self.election_start.read().await;
        if let Some(start) = *election_start {
            start.elapsed() >= self.election_timeout
        } else {
            false
        }
    }

    /// Finalize election and select coordinator
    pub async fn finalize_election(&self) -> Result<String> {
        info!("🏁 Finalizing coordinator election");

        let candidates = self.candidates.read().await;
        if candidates.is_empty() {
            return Err(anyhow!("No candidates available for election"));
        }

        // Find candidate with highest election score
        let winner = candidates
            .values()
            .max_by_key(|c| c.election_score)
            .ok_or_else(|| anyhow!("Failed to determine election winner"))?;

        let winner_id = winner.node_id.clone();
        info!(
            "🎉 Coordinator elected: {} (score: {})",
            winner_id, winner.election_score
        );

        // Set current coordinator
        let mut current = self.current_coordinator.write().await;
        *current = Some(winner_id.clone());

        // Reset election state
        let mut election_start = self.election_start.write().await;
        *election_start = None;

        Ok(winner_id)
    }

    /// Get current coordinator
    pub async fn get_coordinator(&self) -> Option<String> {
        self.current_coordinator.read().await.clone()
    }

    /// Check if this node is the coordinator
    pub async fn is_coordinator(&self) -> bool {
        let coordinator = self.current_coordinator.read().await;
        coordinator.as_ref() == Some(&self.node_id)
    }

    /// Update inference statistics
    pub async fn update_stats(&self, latency_ms: u64) {
        let mut count = self.inference_count.write().await;
        let mut avg_latency = self.average_latency_ms.write().await;

        *count += 1;
        *avg_latency = (*avg_latency * (*count - 1) + latency_ms) / *count;
    }

    /// Get all candidates
    pub async fn get_candidates(&self) -> Vec<ElectionCandidate> {
        self.candidates.read().await.values().cloned().collect()
    }

    /// Remove stale candidates (no heartbeat for 60 seconds)
    pub async fn remove_stale_candidates(&self) -> Result<()> {
        let mut candidates = self.candidates.write().await;
        let now = chrono::Utc::now().timestamp();
        let stale_threshold = 60; // 60 seconds

        let stale_nodes: Vec<String> = candidates
            .iter()
            .filter(|(_, candidate)| {
                now - candidate.last_heartbeat > stale_threshold
            })
            .map(|(node_id, _)| node_id.clone())
            .collect();

        for node_id in &stale_nodes {
            warn!("⚠️ Removing stale candidate: {}", node_id);
            candidates.remove(node_id);
        }

        if !stale_nodes.is_empty() {
            info!("🧹 Removed {} stale candidates", stale_nodes.len());
        }

        Ok(())
    }

    /// Update candidate heartbeat
    pub async fn update_heartbeat(&self, node_id: &str) -> Result<()> {
        let mut candidates = self.candidates.write().await;
        if let Some(candidate) = candidates.get_mut(node_id) {
            candidate.last_heartbeat = chrono::Utc::now().timestamp();
            debug!("💓 Updated heartbeat for candidate: {}", node_id);
        }
        Ok(())
    }

    /// Trigger re-election if coordinator is unavailable
    pub async fn check_coordinator_health(&self) -> Result<bool> {
        let coordinator = self.current_coordinator.read().await;
        if let Some(coord_id) = coordinator.as_ref() {
            let candidates = self.candidates.read().await;
            if let Some(coord_candidate) = candidates.get(coord_id) {
                let now = chrono::Utc::now().timestamp();
                let time_since_heartbeat = now - coord_candidate.last_heartbeat;

                if time_since_heartbeat > 60 {
                    warn!("⚠️ Coordinator {} is unresponsive ({}s since last heartbeat)", coord_id, time_since_heartbeat);
                    return Ok(false);
                }
                return Ok(true);
            }
        }
        // No coordinator set
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_election_score_calculation() {
        let mut candidate = ElectionCandidate {
            node_id: "test-node".to_string(),
            peer_id: "test-peer".to_string(),
            capability: DeviceCapability::CUDA {
                vram_gb: 24,
                compute_capability: "8.6".to_string(),
            },
            uptime_secs: 3600,
            inference_count: 100,
            average_latency_ms: 100,
            last_heartbeat: chrono::Utc::now().timestamp(),
            election_score: 0,
            stake_amount: 0,
        };

        candidate.calculate_election_score();

        // CUDA 24GB = 24000, uptime = 60, latency penalty = 9, reliability = 10
        // Total should be > 24000
        assert!(candidate.election_score > 24000);
        assert!(candidate.election_score < 30000);
    }

    #[tokio::test]
    async fn test_coordinator_election_creation() {
        let capability = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };

        let election = CoordinatorElection::new(
            "node1".to_string(),
            "peer1".to_string(),
            capability,
        );

        assert_eq!(election.node_id, "node1");
        assert!(!election.is_coordinator().await);
    }

    #[tokio::test]
    async fn test_election_workflow() {
        let capability = DeviceCapability::CUDA {
            vram_gb: 12,
            compute_capability: "7.5".to_string(),
        };

        let election = CoordinatorElection::new(
            "node1".to_string(),
            "peer1".to_string(),
            capability,
        );

        // Start election
        election.start_election().await.unwrap();

        // Add another candidate
        let mut candidate2 = ElectionCandidate {
            node_id: "node2".to_string(),
            peer_id: "peer2".to_string(),
            capability: DeviceCapability::CUDA {
                vram_gb: 24,
                compute_capability: "8.6".to_string(),
            },
            uptime_secs: 7200,
            inference_count: 500,
            average_latency_ms: 50,
            last_heartbeat: chrono::Utc::now().timestamp(),
            election_score: 0,
            stake_amount: 0,
        };
        candidate2.calculate_election_score();

        election.add_candidate(candidate2).await.unwrap();

        // Check candidates
        let candidates = election.get_candidates().await;
        assert_eq!(candidates.len(), 2);

        // Finalize election (node2 should win with higher GPU)
        let winner = election.finalize_election().await.unwrap();
        assert_eq!(winner, "node2");

        assert_eq!(election.get_coordinator().await, Some("node2".to_string()));
    }

    #[tokio::test]
    async fn test_stale_candidate_removal() {
        let capability = DeviceCapability::CPU {
            cores: 4,
            ram_gb: 8,
        };

        let election = CoordinatorElection::new(
            "node1".to_string(),
            "peer1".to_string(),
            capability,
        );

        // Add a candidate with old heartbeat
        let old_timestamp = chrono::Utc::now().timestamp() - 120; // 2 minutes ago
        let stale_candidate = ElectionCandidate {
            node_id: "stale-node".to_string(),
            peer_id: "stale-peer".to_string(),
            capability: DeviceCapability::CPU {
                cores: 2,
                ram_gb: 4,
            },
            uptime_secs: 600,
            inference_count: 10,
            average_latency_ms: 200,
            last_heartbeat: old_timestamp,
            election_score: 100,
            stake_amount: 0,
        };

        election.add_candidate(stale_candidate).await.unwrap();
        assert_eq!(election.get_candidates().await.len(), 1);

        // Remove stale candidates
        election.remove_stale_candidates().await.unwrap();
        assert_eq!(election.get_candidates().await.len(), 0);
    }
}
