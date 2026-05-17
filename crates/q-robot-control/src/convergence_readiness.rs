//! Convergence Readiness System - CCC (Conformal Cyclic Cosmology) Integration
//!
//! This module implements Roger Penrose's CCC theory for distributed systems:
//! - **Isolation Phase**: Nodes/robots evolve independently (universe expanding)
//! - **Convergence Phase**: Networks merge, partitions heal (universe contracting)
//! - **Aeon Transition**: System reset/upgrade (conformal rescaling)
//!
//! The k-kristensen parameter determines if entities are "mature enough" to
//! converge peacefully or if convergence becomes conflict.
//!
//! ## DAG-Knight Integration (1000+ nodes)
//!
//! Maps cosmic concepts to practical distributed systems:
//! - **Isolation** → Network partitions, independent evolution
//! - **Convergence** → Partition healing, consensus merge
//! - **K-parameter** → Node maturity, Byzantine resistance
//! - **Transcendence** → Protocol upgrades, fork resolution

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn, error};

use crate::survival_metrics::{SurvivalMetrics, KKristensenParameters};
use crate::blockchain_life::*;

// ============================================================================
// COSMIC PHASE MODELING (CCC Theory)
// ============================================================================

/// Cosmic phase state based on Penrose's Conformal Cyclic Cosmology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmicPhase {
    /// Universe expanding - entities isolated in "garden pots"
    /// Maps to: Network partitions, independent node evolution
    Isolation {
        /// Hubble-like expansion rate (network growth rate)
        expansion_rate: f64,
        /// Time in isolation (blocks since partition)
        isolation_duration: u64,
        /// Number of independent "garden pots" (partitions)
        partition_count: usize,
    },

    /// Universe contracting - convergence beginning
    /// Maps to: Partition healing, consensus synchronization
    Convergence {
        /// Contraction rate (sync speed)
        contraction_rate: f64,
        /// Partitions being merged
        merging_partitions: Vec<PartitionId>,
        /// Estimated blocks until full convergence
        blocks_to_unity: u64,
    },

    /// Conformal reset - aeon transition (Penrose's CCC key moment)
    /// Maps to: Protocol upgrade, hard fork, network reset
    AeonTransition {
        /// Entropy state at transition (0.0 = low/ordered, 1.0 = high/chaotic)
        entropy_state: f64,
        /// Previous aeon's "Hawking points" (historical consensus anchors)
        hawking_points: Vec<ConsensusAnchor>,
        /// New aeon's initial conditions
        new_aeon_params: AeonParameters,
    },

    /// Stable unified state - post-convergence harmony
    /// Maps to: Healthy network, all nodes synchronized
    Harmony {
        /// Collective k-kristensen of the network
        collective_k: f64,
        /// Harmony duration (blocks in unified state)
        harmony_duration: u64,
    },
}

/// Unique identifier for network partitions
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionId(pub String);

/// Consensus anchor point (like Hawking points in CCC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusAnchor {
    pub block_hash: [u8; 32],
    pub height: u64,
    pub validator_signatures: usize,
    pub timestamp: DateTime<Utc>,
}

/// Parameters for a new aeon (protocol version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeonParameters {
    pub protocol_version: String,
    pub consensus_rules: ConsensusRules,
    pub genesis_entropy: f64,
    pub initial_k_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRules {
    pub min_validators: usize,
    pub byzantine_threshold: f64,
    pub finality_depth: u64,
    pub k_requirement: f64,
}

// ============================================================================
// CONVERGENCE READINESS (Core System)
// ============================================================================

/// Convergence Readiness Index - determines if an entity can safely converge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceReadiness {
    /// Entity identifier (robot, node, partition)
    pub entity_id: String,

    /// Current cosmic phase
    pub cosmic_phase: CosmicPhase,

    /// K-kristensen survival parameter
    pub k_kristensen: f64,

    /// Readiness for peaceful convergence (0.0 = predatory, 1.0 = transcendent)
    pub convergence_maturity: f64,

    /// Compatibility scores with other entities
    pub compatibility_matrix: HashMap<String, f64>,

    /// Has transcended competitive/predatory behavior?
    pub transcendence_achieved: bool,

    /// Convergence outcome prediction
    pub predicted_outcome: ConvergenceOutcome,

    /// Time until next phase transition
    pub phase_transition_eta: Option<u64>,

    /// Assessment timestamp
    pub assessed_at: DateTime<Utc>,
}

/// Predicted outcome when entities converge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceOutcome {
    /// Peaceful merge - entities combine harmoniously
    /// Requires: k > 0.9, all parties transcendent
    Communion {
        merged_k: f64,
        synergy_bonus: f64,
    },

    /// Cautious coexistence - maintain boundaries
    /// Requires: k > 0.7, mutual respect
    Observation {
        interaction_distance: f64,
        communication_protocol: String,
    },

    /// Competitive but stable - game-theoretic equilibrium
    /// Requires: k > 0.5, clear rules
    Competition {
        equilibrium_state: String,
        resource_allocation: HashMap<String, f64>,
    },

    /// Conflict - convergence becomes battlefield
    /// Result when: k < 0.5, predatory behavior
    Conflict {
        aggressor_probability: f64,
        expected_casualties: f64,
    },

    /// Absorption - one entity dominates
    /// Result when: large k differential
    Absorption {
        dominant_entity: String,
        absorption_rate: f64,
    },
}

// ============================================================================
// DAG-KNIGHT PRACTICAL USE CASES (1000+ Nodes)
// ============================================================================

/// Practical application of CCC theory to DAG-Knight consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagKnightConvergence {
    /// Network-wide convergence state
    pub network_phase: CosmicPhase,

    /// Per-node k-kristensen scores
    pub node_maturity: HashMap<String, f64>,

    /// Detected network partitions
    pub partitions: Vec<NetworkPartition>,

    /// Partition healing status
    pub healing_operations: Vec<HealingOperation>,

    /// Fork resolution using k-parameter
    pub fork_resolutions: Vec<ForkResolution>,
}

/// Network partition (isolated "garden pot")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartition {
    pub partition_id: PartitionId,
    pub node_ids: Vec<String>,
    pub head_block: [u8; 32],
    pub height: u64,
    pub isolation_start: DateTime<Utc>,
    pub collective_k: f64,
    pub evolutionary_divergence: f64,
}

/// Partition healing operation (convergence event)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingOperation {
    pub operation_id: String,
    pub partitions_involved: Vec<PartitionId>,
    pub healing_strategy: HealingStrategy,
    pub k_requirement: f64,
    pub status: HealingStatus,
    pub blocks_synced: u64,
    pub conflicts_resolved: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealingStrategy {
    /// Peaceful merge - both partitions adopt shared history
    PeacefulMerge { common_ancestor: [u8; 32] },

    /// K-weighted merge - higher k partition takes precedence
    KWeightedMerge { weight_formula: String },

    /// Voting merge - validators vote on canonical chain
    ConsensusVote { quorum_threshold: f64 },

    /// Gradual sync - slow convergence to avoid conflicts
    GradualSync { sync_rate_blocks_per_second: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealingStatus {
    Pending,
    InProgress { progress_percent: f64 },
    Completed { final_height: u64 },
    Failed { reason: String },
}

/// Fork resolution using k-kristensen parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkResolution {
    pub fork_point: u64,
    pub competing_chains: Vec<ChainCandidate>,
    pub resolution_method: ResolutionMethod,
    pub winner: Option<[u8; 32]>,
    pub k_differential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainCandidate {
    pub tip_hash: [u8; 32],
    pub height: u64,
    pub cumulative_k: f64,
    pub validator_support: f64,
    pub age_blocks: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionMethod {
    /// Traditional longest chain
    LongestChain,
    /// Heaviest cumulative k-kristensen
    HeaviestK,
    /// Most validator support weighted by k
    KWeightedVoting,
    /// DAG-Knight GHOST-like selection
    DagKnightGhost,
}

// ============================================================================
// CONVERGENCE CALCULATOR
// ============================================================================

pub struct ConvergenceCalculator {
    /// Threshold for peaceful convergence
    communion_threshold: f64,
    /// Threshold for observation mode
    observation_threshold: f64,
    /// Threshold for competition
    competition_threshold: f64,
    /// Network topology analyzer
    topology_cache: HashMap<String, Vec<String>>,
}

impl ConvergenceCalculator {
    pub fn new() -> Self {
        Self {
            communion_threshold: 0.9,
            observation_threshold: 0.7,
            competition_threshold: 0.5,
            topology_cache: HashMap::new(),
        }
    }

    /// Calculate convergence readiness for an entity
    pub fn assess_readiness(
        &self,
        entity_id: &str,
        k_kristensen: f64,
        survival_metrics: &SurvivalMetrics,
        current_phase: &CosmicPhase,
    ) -> ConvergenceReadiness {
        let convergence_maturity = self.calculate_maturity(k_kristensen, survival_metrics);
        let transcendence = self.check_transcendence(k_kristensen, survival_metrics);
        let predicted_outcome = self.predict_outcome(k_kristensen, convergence_maturity);

        ConvergenceReadiness {
            entity_id: entity_id.to_string(),
            cosmic_phase: current_phase.clone(),
            k_kristensen,
            convergence_maturity,
            compatibility_matrix: HashMap::new(),
            transcendence_achieved: transcendence,
            predicted_outcome,
            phase_transition_eta: self.estimate_phase_transition(current_phase),
            assessed_at: Utc::now(),
        }
    }

    /// Calculate maturity score from k-kristensen and survival metrics
    fn calculate_maturity(&self, k: f64, metrics: &SurvivalMetrics) -> f64 {
        // Maturity combines k-kristensen with evolutionary potential
        let base_maturity = k;
        let evolutionary_bonus = metrics.evolutionary_potential * 0.2;
        let cosmic_resilience_bonus = metrics.cosmic_resilience_score * 0.1;

        (base_maturity + evolutionary_bonus + cosmic_resilience_bonus).min(1.0)
    }

    /// Check if entity has transcended predatory behavior
    fn check_transcendence(&self, k: f64, metrics: &SurvivalMetrics) -> bool {
        // Transcendence requires:
        // 1. High k-kristensen (> 0.85)
        // 2. High cosmic resilience (> 0.8)
        // 3. Positive evolutionary potential (> 0.7)
        k > 0.85
            && metrics.cosmic_resilience_score > 0.8
            && metrics.evolutionary_potential > 0.7
    }

    /// Predict convergence outcome based on k-kristensen
    fn predict_outcome(&self, k: f64, maturity: f64) -> ConvergenceOutcome {
        let combined_score = (k + maturity) / 2.0;

        if combined_score >= self.communion_threshold {
            ConvergenceOutcome::Communion {
                merged_k: k * 1.1, // Synergy bonus
                synergy_bonus: 0.1,
            }
        } else if combined_score >= self.observation_threshold {
            ConvergenceOutcome::Observation {
                interaction_distance: (1.0 - combined_score) * 100.0,
                communication_protocol: "cautious_gossip".to_string(),
            }
        } else if combined_score >= self.competition_threshold {
            ConvergenceOutcome::Competition {
                equilibrium_state: "nash_equilibrium".to_string(),
                resource_allocation: HashMap::new(),
            }
        } else {
            ConvergenceOutcome::Conflict {
                aggressor_probability: 1.0 - combined_score,
                expected_casualties: (1.0 - combined_score) * 0.5,
            }
        }
    }

    /// Estimate blocks until phase transition
    fn estimate_phase_transition(&self, phase: &CosmicPhase) -> Option<u64> {
        match phase {
            CosmicPhase::Isolation { expansion_rate: _, isolation_duration, .. } => {
                // Estimate when isolation ends (network reconnects)
                let natural_reconnect_time: u64 = 1000; // blocks
                Some(natural_reconnect_time.saturating_sub(*isolation_duration))
            }
            CosmicPhase::Convergence { blocks_to_unity, .. } => {
                Some(*blocks_to_unity)
            }
            CosmicPhase::AeonTransition { .. } => {
                Some(10) // Quick transition
            }
            CosmicPhase::Harmony { .. } => {
                None // Stable state
            }
        }
    }

    /// Calculate compatibility between two entities
    pub fn calculate_compatibility(
        &self,
        entity_a: &ConvergenceReadiness,
        entity_b: &ConvergenceReadiness,
    ) -> f64 {
        // Compatibility based on k-kristensen similarity and transcendence
        let k_similarity = 1.0 - (entity_a.k_kristensen - entity_b.k_kristensen).abs();
        let maturity_similarity = 1.0 - (entity_a.convergence_maturity - entity_b.convergence_maturity).abs();
        let transcendence_bonus = if entity_a.transcendence_achieved && entity_b.transcendence_achieved {
            0.2
        } else {
            0.0
        };

        (k_similarity * 0.4 + maturity_similarity * 0.4 + transcendence_bonus).min(1.0)
    }
}

// ============================================================================
// DAG-KNIGHT PRACTICAL USE CASE IMPLEMENTATIONS
// ============================================================================

impl DagKnightConvergence {
    pub fn new() -> Self {
        Self {
            network_phase: CosmicPhase::Harmony {
                collective_k: 0.8,
                harmony_duration: 0,
            },
            node_maturity: HashMap::new(),
            partitions: Vec::new(),
            healing_operations: Vec::new(),
            fork_resolutions: Vec::new(),
        }
    }

    /// USE CASE 1: Network Partition Detection & Healing
    /// When network splits into isolated groups, track their independent evolution
    /// and manage the convergence when they reconnect
    pub fn detect_partition(&mut self, partition_nodes: Vec<String>, head_block: [u8; 32], height: u64) {
        let partition_id = PartitionId(format!("partition_{}", height));

        // Calculate collective k for the partition
        let collective_k: f64 = partition_nodes.iter()
            .filter_map(|id| self.node_maturity.get(id))
            .sum::<f64>() / partition_nodes.len() as f64;

        let partition = NetworkPartition {
            partition_id: partition_id.clone(),
            node_ids: partition_nodes,
            head_block,
            height,
            isolation_start: Utc::now(),
            collective_k,
            evolutionary_divergence: 0.0,
        };

        self.partitions.push(partition);

        // Update network phase to Isolation
        self.network_phase = CosmicPhase::Isolation {
            expansion_rate: 0.01,
            isolation_duration: 0,
            partition_count: self.partitions.len(),
        };

        info!(
            "🌌 [CCC] Network partition detected: {} nodes isolated, collective k={:.4}",
            self.partitions.last().map(|p| p.node_ids.len()).unwrap_or(0),
            collective_k
        );
    }

    /// USE CASE 2: K-Weighted Fork Resolution
    /// Resolve competing chains using k-kristensen as tiebreaker
    pub fn resolve_fork(&mut self, fork_point: u64, chains: Vec<ChainCandidate>) -> Option<[u8; 32]> {
        if chains.is_empty() {
            return None;
        }

        // Calculate k-weighted score for each chain
        let mut scored_chains: Vec<(f64, &ChainCandidate)> = chains.iter()
            .map(|chain| {
                let k_score = chain.cumulative_k * 0.4;
                let height_score = (chain.height - fork_point) as f64 * 0.3;
                let validator_score = chain.validator_support * 0.3;
                (k_score + height_score + validator_score, chain)
            })
            .collect();

        scored_chains.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let winner = scored_chains.first().map(|(_, chain)| chain.tip_hash);
        let k_differential = if scored_chains.len() >= 2 {
            scored_chains[0].0 - scored_chains[1].0
        } else {
            1.0
        };

        let resolution = ForkResolution {
            fork_point,
            competing_chains: chains,
            resolution_method: ResolutionMethod::KWeightedVoting,
            winner,
            k_differential,
        };

        self.fork_resolutions.push(resolution);

        info!(
            "⚔️ [CCC] Fork resolved at height {}: k-differential={:.4}",
            fork_point, k_differential
        );

        winner
    }

    /// USE CASE 3: Partition Healing (Convergence)
    /// Merge partitions when they reconnect, using k-kristensen to guide strategy
    pub fn initiate_healing(&mut self, partition_ids: Vec<PartitionId>) -> Result<String> {
        let partitions: Vec<&NetworkPartition> = self.partitions.iter()
            .filter(|p| partition_ids.contains(&p.partition_id))
            .collect();

        if partitions.len() < 2 {
            return Err(anyhow::anyhow!("Need at least 2 partitions to heal"));
        }

        // Calculate collective k for all involved partitions
        let collective_k: f64 = partitions.iter()
            .map(|p| p.collective_k)
            .sum::<f64>() / partitions.len() as f64;

        // Choose healing strategy based on k-kristensen
        let strategy = if collective_k > 0.9 {
            // High k = peaceful merge
            HealingStrategy::PeacefulMerge {
                common_ancestor: partitions[0].head_block,
            }
        } else if collective_k > 0.7 {
            // Medium k = k-weighted merge
            HealingStrategy::KWeightedMerge {
                weight_formula: "cumulative_k * validator_support".to_string(),
            }
        } else if collective_k > 0.5 {
            // Low-medium k = voting
            HealingStrategy::ConsensusVote {
                quorum_threshold: 0.67,
            }
        } else {
            // Low k = gradual sync to avoid conflicts
            HealingStrategy::GradualSync {
                sync_rate_blocks_per_second: 10.0,
            }
        };

        let operation_id = format!("heal_{}", Utc::now().timestamp());

        let healing = HealingOperation {
            operation_id: operation_id.clone(),
            partitions_involved: partition_ids,
            healing_strategy: strategy,
            k_requirement: collective_k,
            status: HealingStatus::InProgress { progress_percent: 0.0 },
            blocks_synced: 0,
            conflicts_resolved: 0,
        };

        self.healing_operations.push(healing);

        // Update network phase to Convergence
        self.network_phase = CosmicPhase::Convergence {
            contraction_rate: collective_k,
            merging_partitions: self.partitions.iter().map(|p| p.partition_id.clone()).collect(),
            blocks_to_unity: 1000,
        };

        info!(
            "🌀 [CCC] Partition healing initiated: collective k={:.4}, strategy={:?}",
            collective_k, self.healing_operations.last().map(|h| &h.healing_strategy)
        );

        Ok(operation_id)
    }

    /// USE CASE 4: Node Maturity Assessment for Validator Selection
    /// Select validators based on k-kristensen for Byzantine resistance
    pub fn select_validators(&self, required_count: usize) -> Vec<String> {
        let mut nodes: Vec<(&String, &f64)> = self.node_maturity.iter().collect();
        nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        nodes.iter()
            .take(required_count)
            .map(|(id, _)| (*id).clone())
            .collect()
    }

    /// USE CASE 5: Protocol Upgrade (Aeon Transition)
    /// Manage hard fork using CCC aeon transition model
    pub fn initiate_aeon_transition(&mut self, new_version: &str, activation_height: u64) -> AeonParameters {
        // Capture current state as "Hawking points"
        let hawking_points: Vec<ConsensusAnchor> = self.fork_resolutions.iter()
            .filter_map(|fr| fr.winner.map(|hash| ConsensusAnchor {
                block_hash: hash,
                height: fr.fork_point,
                validator_signatures: fr.competing_chains.len(),
                timestamp: Utc::now(),
            }))
            .collect();

        // Calculate new aeon parameters
        let collective_k: f64 = self.node_maturity.values().sum::<f64>()
            / self.node_maturity.len().max(1) as f64;

        let new_params = AeonParameters {
            protocol_version: new_version.to_string(),
            consensus_rules: ConsensusRules {
                min_validators: 21,
                byzantine_threshold: 0.33,
                finality_depth: 6,
                k_requirement: collective_k * 0.9, // Slightly lower than current for growth
            },
            genesis_entropy: 0.1, // Low entropy = ordered new beginning
            initial_k_threshold: 0.5,
        };

        self.network_phase = CosmicPhase::AeonTransition {
            entropy_state: 0.1,
            hawking_points,
            new_aeon_params: new_params.clone(),
        };

        info!(
            "🌟 [CCC] Aeon transition initiated: {} → {}, activation at height {}",
            "current", new_version, activation_height
        );

        new_params
    }

    /// USE CASE 6: Swarm Coordination for Water Robots
    /// Coordinate water robot swarms based on convergence readiness
    pub fn coordinate_swarm(
        &self,
        robot_ids: &[String],
        mission_type: &str,
    ) -> SwarmCoordinationPlan {
        // Group robots by k-kristensen maturity
        let mut mature_robots = Vec::new();
        let mut developing_robots = Vec::new();
        let mut immature_robots = Vec::new();

        for id in robot_ids {
            if let Some(&k) = self.node_maturity.get(id) {
                if k > 0.8 {
                    mature_robots.push(id.clone());
                } else if k > 0.5 {
                    developing_robots.push(id.clone());
                } else {
                    immature_robots.push(id.clone());
                }
            }
        }

        // Assign roles based on maturity
        SwarmCoordinationPlan {
            mission_type: mission_type.to_string(),
            leaders: mature_robots,        // High k = leadership roles
            workers: developing_robots,    // Medium k = task execution
            learners: immature_robots,     // Low k = learning/support
            coordination_strategy: match &self.network_phase {
                CosmicPhase::Harmony { .. } => "full_swarm_sync".to_string(),
                CosmicPhase::Convergence { .. } => "gradual_merge".to_string(),
                CosmicPhase::Isolation { .. } => "independent_cells".to_string(),
                CosmicPhase::AeonTransition { .. } => "hold_position".to_string(),
            },
        }
    }
}

/// Swarm coordination plan based on convergence state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCoordinationPlan {
    pub mission_type: String,
    pub leaders: Vec<String>,
    pub workers: Vec<String>,
    pub learners: Vec<String>,
    pub coordination_strategy: String,
}

// ============================================================================
// PRACTICAL USE CASES SUMMARY (1000+ Nodes)
// ============================================================================

/// Summary of practical use cases for DAG-Knight with 1000+ nodes
pub fn describe_use_cases() -> Vec<UseCaseDescription> {
    vec![
        UseCaseDescription {
            name: "Partition-Tolerant Consensus".to_string(),
            description: "Handle network splits gracefully using CCC isolation/convergence model. \
                         Partitions evolve independently, then merge using k-weighted consensus.".to_string(),
            k_requirement: 0.6,
            nodes_required: 100,
            real_world_application: "Geo-distributed blockchain with intermittent connectivity".to_string(),
        },
        UseCaseDescription {
            name: "K-Weighted Validator Selection".to_string(),
            description: "Select validators based on k-kristensen maturity scores. \
                         Higher k = more reliable, Byzantine-resistant validators.".to_string(),
            k_requirement: 0.8,
            nodes_required: 21,
            real_world_application: "DPoS-style consensus with quality-based delegation".to_string(),
        },
        UseCaseDescription {
            name: "Graceful Protocol Upgrades".to_string(),
            description: "Manage hard forks using aeon transition model. \
                         Capture consensus anchors (Hawking points) for continuity.".to_string(),
            k_requirement: 0.7,
            nodes_required: 500,
            real_world_application: "Zero-downtime blockchain upgrades".to_string(),
        },
        UseCaseDescription {
            name: "Fork Resolution with K-Parameter".to_string(),
            description: "Resolve competing chains using k-kristensen as tiebreaker. \
                         Favors chains with more mature/reliable validators.".to_string(),
            k_requirement: 0.5,
            nodes_required: 50,
            real_world_application: "Finality gadget for probabilistic consensus".to_string(),
        },
        UseCaseDescription {
            name: "Swarm Intelligence Coordination".to_string(),
            description: "Coordinate 1000+ water robots based on maturity levels. \
                         Mature robots lead, developing robots execute, immature robots learn.".to_string(),
            k_requirement: 0.3,
            nodes_required: 1000,
            real_world_application: "Ocean monitoring, coral restoration, search & rescue".to_string(),
        },
        UseCaseDescription {
            name: "Byzantine Fault Tolerance Enhancement".to_string(),
            description: "Use k-parameter to identify and isolate Byzantine nodes. \
                         Nodes with declining k are quarantined before causing damage.".to_string(),
            k_requirement: 0.9,
            nodes_required: 100,
            real_world_application: "High-security financial networks".to_string(),
        },
        UseCaseDescription {
            name: "Cosmic-Scale Data Persistence".to_string(),
            description: "Model data survival across cosmic timescales. \
                         High-k nodes store critical data for universe-scale resilience.".to_string(),
            k_requirement: 0.95,
            nodes_required: 1000,
            real_world_application: "Long-term archival, civilizational memory storage".to_string(),
        },
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseCaseDescription {
    pub name: String,
    pub description: String,
    pub k_requirement: f64,
    pub nodes_required: usize,
    pub real_world_application: String,
}

// ============================================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================================

impl std::fmt::Display for CosmicPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CosmicPhase::Isolation { partition_count, .. } => {
                write!(f, "🌌 Isolation ({} partitions)", partition_count)
            }
            CosmicPhase::Convergence { blocks_to_unity, .. } => {
                write!(f, "🌀 Convergence ({} blocks to unity)", blocks_to_unity)
            }
            CosmicPhase::AeonTransition { entropy_state, .. } => {
                write!(f, "🌟 Aeon Transition (entropy: {:.2})", entropy_state)
            }
            CosmicPhase::Harmony { collective_k, .. } => {
                write!(f, "☮️ Harmony (k: {:.4})", collective_k)
            }
        }
    }
}

impl std::fmt::Display for ConvergenceOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvergenceOutcome::Communion { merged_k, .. } => {
                write!(f, "🤝 Communion (merged k: {:.4})", merged_k)
            }
            ConvergenceOutcome::Observation { interaction_distance, .. } => {
                write!(f, "👁️ Observation (distance: {:.1})", interaction_distance)
            }
            ConvergenceOutcome::Competition { equilibrium_state, .. } => {
                write!(f, "⚖️ Competition ({})", equilibrium_state)
            }
            ConvergenceOutcome::Conflict { aggressor_probability, .. } => {
                write!(f, "⚔️ Conflict (aggression: {:.1}%)", aggressor_probability * 100.0)
            }
            ConvergenceOutcome::Absorption { dominant_entity, .. } => {
                write!(f, "🕳️ Absorption (by {})", dominant_entity)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_outcome_prediction() {
        let calc = ConvergenceCalculator::new();

        // High k should predict communion
        let outcome_high = calc.predict_outcome(0.95, 0.92);
        assert!(matches!(outcome_high, ConvergenceOutcome::Communion { .. }));

        // Medium k should predict observation
        let outcome_med = calc.predict_outcome(0.75, 0.72);
        assert!(matches!(outcome_med, ConvergenceOutcome::Observation { .. }));

        // Low k should predict conflict
        let outcome_low = calc.predict_outcome(0.3, 0.25);
        assert!(matches!(outcome_low, ConvergenceOutcome::Conflict { .. }));
    }

    #[test]
    fn test_use_cases_coverage() {
        let use_cases = describe_use_cases();
        assert!(use_cases.len() >= 5, "Should have multiple practical use cases");

        // Ensure we have both small and large node requirements
        let max_nodes = use_cases.iter().map(|u| u.nodes_required).max().unwrap();
        assert!(max_nodes >= 1000, "Should have use case for 1000+ nodes");
    }
}
