/// Network View Consistency Checker for Q-NarwhalKnight
/// Ensures all validators have a consistent view of network topology
use anyhow::Result;
use q_types::ValidatorId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::dag_sync::{DagStateSummary, DagSyncManager};
use super::network_manager::NetworkManager;
use super::peer_registry::PeerRegistry;

/// Network view consistency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    pub check_timestamp: u64,
    pub local_validator_id: ValidatorId,
    pub network_view_hash: Option<[u8; 32]>,
    pub peer_comparisons: Vec<PeerConsistencyComparison>,
    pub inconsistencies: Vec<NetworkInconsistency>,
    pub consistency_score: f64,
    pub recommendation: ConsistencyRecommendation,
}

/// Comparison with a specific peer's network view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConsistencyComparison {
    pub peer_id: ValidatorId,
    pub peer_network_hash: Option<[u8; 32]>,
    pub dag_state_hash: [u8; 32],
    pub round_difference: i64,
    pub peer_count_difference: i64,
    pub is_consistent: bool,
}

/// Types of network inconsistencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkInconsistency {
    NetworkViewMismatch {
        peer: ValidatorId,
        local_hash: [u8; 32],
        peer_hash: [u8; 32],
    },
    DagStateMismatch {
        peer: ValidatorId,
        round_gap: u64,
        vertex_count_difference: i64,
    },
    PeerCountMismatch {
        peer: ValidatorId,
        local_peer_count: usize,
        peer_peer_count: usize,
    },
    StaleNetworkView {
        peer: ValidatorId,
        last_seen_age: Duration,
    },
}

/// Recommendations based on consistency check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyRecommendation {
    NetworkHealthy,
    MinorInconsistenciesDetected,
    MajorSyncRequired,
    NetworkPartitionSuspected,
    PeerIsolationDetected { isolated_peers: Vec<ValidatorId> },
}

/// Network view consistency checker
pub struct ConsistencyChecker {
    local_validator_id: ValidatorId,
    check_history: RwLock<Vec<ConsistencyReport>>,
    max_history_size: usize,
    consistency_threshold: f64,
}

impl ConsistencyChecker {
    pub fn new(local_validator_id: ValidatorId) -> Self {
        Self {
            local_validator_id,
            check_history: RwLock::new(Vec::new()),
            max_history_size: 100,
            consistency_threshold: 0.9, // 90% consistency threshold
        }
    }

    /// Perform comprehensive network consistency check
    pub async fn check_network_consistency(
        &self,
        network_manager: &NetworkManager,
    ) -> Result<ConsistencyReport> {
        info!("🔍 Performing network consistency check");

        let start_time = Instant::now();
        
        // Get local network view
        let local_network_hash = network_manager.get_network_view_hash().await;
        let local_stats = network_manager.get_network_stats().await;
        let connected_peers = network_manager.get_connected_peer_info().await;

        // Compare with each peer
        let mut peer_comparisons = Vec::new();
        let mut inconsistencies = Vec::new();

        for peer in &connected_peers {
            let comparison = self.compare_with_peer(peer, &local_stats, local_network_hash).await?;
            
            if !comparison.is_consistent {
                inconsistencies.extend(self.identify_inconsistencies(&comparison).await);
            }
            
            peer_comparisons.push(comparison);
        }

        // Calculate overall consistency score
        let consistency_score = self.calculate_consistency_score(&peer_comparisons);
        
        // Generate recommendation
        let recommendation = self.generate_recommendation(consistency_score, &inconsistencies);

        let report = ConsistencyReport {
            check_timestamp: chrono::Utc::now().timestamp() as u64,
            local_validator_id: self.local_validator_id,
            network_view_hash: local_network_hash,
            peer_comparisons,
            inconsistencies,
            consistency_score,
            recommendation,
        };

        // Store in history
        self.store_report(report.clone()).await;

        let duration = start_time.elapsed();
        info!(
            "✅ Consistency check completed in {}ms - Score: {:.1}%",
            duration.as_millis(),
            consistency_score * 100.0
        );

        Ok(report)
    }

    /// Compare network view with a specific peer
    async fn compare_with_peer(
        &self,
        peer: &super::peer_registry::PeerInfo,
        local_stats: &super::network_manager::NetworkManagerStats,
        local_network_hash: Option<[u8; 32]>,
    ) -> Result<PeerConsistencyComparison> {
        debug!("🔍 Comparing network view with peer {}", hex::encode(peer.validator_id));

        // In a real implementation, this would request the peer's network state
        // For now, simulate the comparison
        let peer_network_hash = Some([1u8; 32]); // Mock peer hash
        let peer_dag_hash = [2u8; 32]; // Mock DAG state hash
        let peer_round = local_stats.dag_syncs_performed + 1; // Simulate slight difference
        let peer_peer_count = local_stats.connected_peers; // Assume same for now

        let round_difference = peer_round as i64 - local_stats.dag_syncs_performed as i64;
        let peer_count_difference = peer_peer_count as i64 - local_stats.connected_peers as i64;

        let is_consistent = match (local_network_hash, peer_network_hash) {
            (Some(local), Some(peer_hash)) => {
                local == peer_hash && 
                round_difference.abs() <= 2 && 
                peer_count_difference.abs() <= 1
            }
            _ => false,
        };

        Ok(PeerConsistencyComparison {
            peer_id: peer.validator_id,
            peer_network_hash,
            dag_state_hash: peer_dag_hash,
            round_difference,
            peer_count_difference,
            is_consistent,
        })
    }

    /// Identify specific inconsistencies from comparison
    async fn identify_inconsistencies(
        &self,
        comparison: &PeerConsistencyComparison,
    ) -> Vec<NetworkInconsistency> {
        let mut inconsistencies = Vec::new();

        // Network view hash mismatch
        if let Some(peer_hash) = comparison.peer_network_hash {
            inconsistencies.push(NetworkInconsistency::NetworkViewMismatch {
                peer: comparison.peer_id,
                local_hash: [0u8; 32], // Would be actual local hash
                peer_hash,
            });
        }

        // Significant round difference
        if comparison.round_difference.abs() > 5 {
            inconsistencies.push(NetworkInconsistency::DagStateMismatch {
                peer: comparison.peer_id,
                round_gap: comparison.round_difference.abs() as u64,
                vertex_count_difference: comparison.peer_count_difference,
            });
        }

        // Peer count mismatch
        if comparison.peer_count_difference.abs() > 2 {
            inconsistencies.push(NetworkInconsistency::PeerCountMismatch {
                peer: comparison.peer_id,
                local_peer_count: 0, // Would be actual count
                peer_peer_count: 0, // Would be actual count
            });
        }

        inconsistencies
    }

    /// Calculate overall consistency score (0.0 to 1.0)
    fn calculate_consistency_score(&self, comparisons: &[PeerConsistencyComparison]) -> f64 {
        if comparisons.is_empty() {
            return 1.0; // Perfect consistency with no peers (isolated)
        }

        let consistent_peers = comparisons.iter().filter(|c| c.is_consistent).count();
        consistent_peers as f64 / comparisons.len() as f64
    }

    /// Generate recommendation based on consistency analysis
    fn generate_recommendation(
        &self,
        consistency_score: f64,
        inconsistencies: &[NetworkInconsistency],
    ) -> ConsistencyRecommendation {
        if consistency_score >= 0.95 {
            ConsistencyRecommendation::NetworkHealthy
        } else if consistency_score >= 0.8 {
            ConsistencyRecommendation::MinorInconsistenciesDetected
        } else if consistency_score >= 0.5 {
            ConsistencyRecommendation::MajorSyncRequired
        } else {
            // Check for potential network partition
            let dag_mismatches: Vec<_> = inconsistencies
                .iter()
                .filter_map(|inc| match inc {
                    NetworkInconsistency::DagStateMismatch { peer, round_gap, .. } 
                        if *round_gap > 10 => Some(*peer),
                    _ => None,
                })
                .collect();

            if dag_mismatches.len() > inconsistencies.len() / 2 {
                ConsistencyRecommendation::NetworkPartitionSuspected
            } else if !dag_mismatches.is_empty() {
                ConsistencyRecommendation::PeerIsolationDetected {
                    isolated_peers: dag_mismatches,
                }
            } else {
                ConsistencyRecommendation::MajorSyncRequired
            }
        }
    }

    /// Store consistency report in history
    async fn store_report(&self, report: ConsistencyReport) {
        let mut history = self.check_history.write().await;
        
        history.push(report);
        
        // Limit history size
        if history.len() > self.max_history_size {
            history.remove(0);
        }
    }

    /// Get consistency check history
    pub async fn get_consistency_history(&self) -> Vec<ConsistencyReport> {
        self.check_history.read().await.clone()
    }

    /// Get consistency trends over time
    pub async fn get_consistency_trends(&self) -> ConsistencyTrends {
        let history = self.check_history.read().await;
        
        if history.is_empty() {
            return ConsistencyTrends::default();
        }

        let scores: Vec<f64> = history.iter().map(|r| r.consistency_score).collect();
        let recent_scores: Vec<f64> = scores.iter().rev().take(10).cloned().collect();
        
        let average_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let recent_average = if recent_scores.is_empty() { 
            0.0 
        } else { 
            recent_scores.iter().sum::<f64>() / recent_scores.len() as f64 
        };
        
        let trend_direction = if recent_average > average_score + 0.05 {
            TrendDirection::Improving
        } else if recent_average < average_score - 0.05 {
            TrendDirection::Deteriorating
        } else {
            TrendDirection::Stable
        };

        ConsistencyTrends {
            average_consistency_score: average_score,
            recent_average_score: recent_average,
            trend_direction,
            total_checks: history.len(),
            inconsistency_frequency: self.calculate_inconsistency_frequency(&history),
        }
    }

    /// Calculate frequency of inconsistencies
    fn calculate_inconsistency_frequency(&self, history: &[ConsistencyReport]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let reports_with_inconsistencies = history
            .iter()
            .filter(|r| !r.inconsistencies.is_empty())
            .count();

        reports_with_inconsistencies as f64 / history.len() as f64
    }

    /// Check if network needs immediate attention
    pub async fn needs_immediate_attention(&self) -> bool {
        let history = self.get_consistency_history().await;
        
        if history.len() < 3 {
            return false;
        }

        // Check last 3 reports
        let recent_reports: Vec<_> = history.iter().rev().take(3).collect();
        
        // If consistency score has been below threshold for 3 consecutive checks
        recent_reports
            .iter()
            .all(|r| r.consistency_score < self.consistency_threshold)
    }
}

/// Consistency trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyTrends {
    pub average_consistency_score: f64,
    pub recent_average_score: f64,
    pub trend_direction: TrendDirection,
    pub total_checks: usize,
    pub inconsistency_frequency: f64,
}

impl Default for ConsistencyTrends {
    fn default() -> Self {
        Self {
            average_consistency_score: 1.0,
            recent_average_score: 1.0,
            trend_direction: TrendDirection::Stable,
            total_checks: 0,
            inconsistency_frequency: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Deteriorating,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consistency_checker_creation() {
        let validator_id = [1u8; 32];
        let checker = ConsistencyChecker::new(validator_id);
        
        let history = checker.get_consistency_history().await;
        assert_eq!(history.len(), 0);
    }

    #[tokio::test]
    async fn test_consistency_score_calculation() {
        let validator_id = [1u8; 32];
        let checker = ConsistencyChecker::new(validator_id);
        
        let comparisons = vec![
            PeerConsistencyComparison {
                peer_id: [2u8; 32],
                peer_network_hash: Some([1u8; 32]),
                dag_state_hash: [2u8; 32],
                round_difference: 0,
                peer_count_difference: 0,
                is_consistent: true,
            },
            PeerConsistencyComparison {
                peer_id: [3u8; 32],
                peer_network_hash: Some([2u8; 32]),
                dag_state_hash: [3u8; 32],
                round_difference: 10,
                peer_count_difference: 5,
                is_consistent: false,
            },
        ];
        
        let score = checker.calculate_consistency_score(&comparisons);
        assert_eq!(score, 0.5); // 1 out of 2 peers consistent
    }

    #[tokio::test]
    async fn test_trend_analysis() {
        let validator_id = [1u8; 32];
        let checker = ConsistencyChecker::new(validator_id);
        
        // Simulate some consistency reports
        let reports = vec![
            ConsistencyReport {
                check_timestamp: 1000,
                local_validator_id: validator_id,
                network_view_hash: Some([1u8; 32]),
                peer_comparisons: vec![],
                inconsistencies: vec![],
                consistency_score: 0.9,
                recommendation: ConsistencyRecommendation::NetworkHealthy,
            },
            ConsistencyReport {
                check_timestamp: 2000,
                local_validator_id: validator_id,
                network_view_hash: Some([1u8; 32]),
                peer_comparisons: vec![],
                inconsistencies: vec![],
                consistency_score: 0.95,
                recommendation: ConsistencyRecommendation::NetworkHealthy,
            },
        ];
        
        {
            let mut history = checker.check_history.write().await;
            history.extend(reports);
        }
        
        let trends = checker.get_consistency_trends().await;
        assert_eq!(trends.total_checks, 2);
        assert!(trends.average_consistency_score > 0.9);
    }
}