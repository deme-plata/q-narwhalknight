//! On-chain Architecture Evolution via NAS Governance
//!
//! Decentralized neural architecture search where validators propose, vote on,
//! and evolve model architectures through stake-weighted governance. Combines
//! genetic algorithms with blockchain consensus for autonomous model improvement.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    NAS GOVERNANCE INFRASTRUCTURE                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
//! │  │  PROPOSAL PHASE │   │  VOTING PHASE   │   │ EXECUTION PHASE │           │
//! │  │                 │   │                 │   │                 │           │
//! │  │ • Submit arch   │──►│ • Stake-weight  │──►│ • Deploy winner │           │
//! │  │ • Fitness eval  │   │ • Quadratic     │   │ • Update global │           │
//! │  │ • Bond deposit  │   │ • Time-locked   │   │ • Reward/slash  │           │
//! │  └─────────────────┘   └─────────────────┘   └─────────────────┘           │
//! │           │                    │                      │                     │
//! │           ▼                    ▼                      ▼                     │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                    ON-CHAIN STATE                                │       │
//! │  │  • Active architecture (canonical model)                        │       │
//! │  │  • Proposal queue (pending architectures)                       │       │
//! │  │  • Vote tally (stake-weighted counts)                           │       │
//! │  │  • History (past architectures + performance)                   │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! │           │                    │                      │                     │
//! │           ▼                    ▼                      ▼                     │
//! │  ┌─────────────────────────────────────────────────────────────────┐       │
//! │  │                  EVOLUTIONARY OPERATORS                          │       │
//! │  │  • Mutation: Random layer changes (from NAS engine)             │       │
//! │  │  • Crossover: Combine successful architectures                   │       │
//! │  │  • Selection: Tournament based on on-chain metrics              │       │
//! │  └─────────────────────────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use sha3::{Sha3_256, Digest};
use tracing::{info, warn, error};

use crate::evolution::{Architecture, LayerType, FitnessScore, NasEngine, NasConfig};

/// Governance configuration
#[derive(Clone, Debug)]
pub struct GovernanceConfig {
    /// Minimum stake to submit proposal
    pub min_proposal_stake: u64,
    /// Proposal bond (slashed if proposal fails badly)
    pub proposal_bond: u64,
    /// Voting period duration (blocks)
    pub voting_period_blocks: u64,
    /// Execution delay after vote passes (blocks)
    pub execution_delay_blocks: u64,
    /// Quorum percentage (0-100)
    pub quorum_percentage: u32,
    /// Approval threshold percentage (0-100)
    pub approval_threshold: u32,
    /// Maximum concurrent proposals
    pub max_concurrent_proposals: usize,
    /// Fitness improvement threshold for auto-approval
    pub auto_approve_fitness_delta: f64,
    /// Enable quadratic voting
    pub quadratic_voting: bool,
    /// Minimum performance period before evaluation (blocks)
    pub performance_eval_period: u64,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            min_proposal_stake: 100_000_000_000,   // 1000 QUG
            proposal_bond: 10_000_000_000,         // 100 QUG
            voting_period_blocks: 1000,            // ~1 day at 10 TPS
            execution_delay_blocks: 100,
            quorum_percentage: 20,
            approval_threshold: 60,
            max_concurrent_proposals: 5,
            auto_approve_fitness_delta: 0.15,      // 15% improvement
            quadratic_voting: true,
            performance_eval_period: 5000,
        }
    }
}

/// Architecture proposal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureProposal {
    /// Unique proposal ID
    pub id: u64,
    /// Proposer address
    pub proposer: [u8; 32],
    /// Proposed architecture
    pub architecture: Architecture,
    /// Bonded amount
    pub bond: u64,
    /// Submission block height
    pub submitted_at: u64,
    /// Proposal status
    pub status: ProposalStatus,
    /// Preliminary fitness (from off-chain evaluation)
    pub preliminary_fitness: Option<FitnessScore>,
    /// On-chain performance metrics (after deployment)
    pub on_chain_metrics: Option<OnChainMetrics>,
    /// Description/rationale
    pub description: String,
    /// Parent proposal (if evolved from another)
    pub parent_id: Option<u64>,
    /// Hash of architecture for deduplication
    pub architecture_hash: [u8; 32],
}

/// Proposal status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    /// Awaiting voting
    Pending,
    /// Currently in voting period
    Voting { end_block: u64 },
    /// Approved, waiting execution
    Approved { execute_at: u64 },
    /// Rejected by vote
    Rejected,
    /// Successfully deployed
    Deployed { deployed_at: u64 },
    /// Failed performance threshold
    Failed { reason: String },
    /// Withdrawn by proposer
    Withdrawn,
}

/// Vote on a proposal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    /// Voter address
    pub voter: [u8; 32],
    /// Proposal ID
    pub proposal_id: u64,
    /// Vote direction
    pub direction: VoteDirection,
    /// Voting power (stake-weighted, possibly quadratic)
    pub voting_power: u64,
    /// Vote timestamp
    pub timestamp: u64,
    /// Optional rationale
    pub rationale: Option<String>,
}

/// Vote direction
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteDirection {
    For,
    Against,
    Abstain,
}

/// On-chain performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OnChainMetrics {
    /// Prediction accuracy (0-1)
    pub accuracy: f64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Total predictions made
    pub total_predictions: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Revenue generated (QUG)
    pub revenue_generated: u64,
    /// Gas/compute cost
    pub compute_cost: u64,
    /// Start block
    pub start_block: u64,
    /// End block
    pub end_block: u64,
}

/// Architecture governance state
#[derive(Clone, Debug)]
pub struct GovernanceState {
    /// Current active architecture
    pub active_architecture: Option<Architecture>,
    /// Active architecture ID
    pub active_id: Option<u64>,
    /// Next proposal ID
    pub next_proposal_id: u64,
    /// Total proposals submitted
    pub total_proposals: u64,
    /// Total votes cast
    pub total_votes: u64,
}

/// NAS Governance Coordinator
pub struct NasGovernance {
    /// Configuration
    config: GovernanceConfig,
    /// Current state
    state: Arc<RwLock<GovernanceState>>,
    /// Proposal storage
    proposals: Arc<RwLock<HashMap<u64, ArchitectureProposal>>>,
    /// Votes storage
    votes: Arc<RwLock<HashMap<u64, Vec<Vote>>>>,
    /// Validator stakes
    stakes: Arc<RwLock<HashMap<[u8; 32], u64>>>,
    /// NAS engine for fitness evaluation
    nas_engine: Arc<RwLock<NasEngine>>,
    /// Architecture history (id -> performance)
    history: Arc<RwLock<BTreeMap<u64, ArchitecturePerformance>>>,
    /// Current block height
    current_block: Arc<RwLock<u64>>,
}

/// Historical architecture performance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitecturePerformance {
    pub architecture_id: u64,
    pub architecture_hash: [u8; 32],
    pub deployment_block: u64,
    pub retirement_block: Option<u64>,
    pub metrics: OnChainMetrics,
    pub fitness: FitnessScore,
}

impl NasGovernance {
    /// Create new NAS governance coordinator
    pub fn new(config: GovernanceConfig) -> Self {
        info!("🏛️ Initializing NAS Governance");
        info!("   Min proposal stake: {} QUG", config.min_proposal_stake / 100_000_000);
        info!("   Voting period: {} blocks", config.voting_period_blocks);
        info!("   Quorum: {}%", config.quorum_percentage);

        let nas_config = NasConfig::default();
        let nas_engine = NasEngine::new(nas_config);

        Self {
            config,
            state: Arc::new(RwLock::new(GovernanceState {
                active_architecture: None,
                active_id: None,
                next_proposal_id: 1,
                total_proposals: 0,
                total_votes: 0,
            })),
            proposals: Arc::new(RwLock::new(HashMap::new())),
            votes: Arc::new(RwLock::new(HashMap::new())),
            stakes: Arc::new(RwLock::new(HashMap::new())),
            nas_engine: Arc::new(RwLock::new(nas_engine)),
            history: Arc::new(RwLock::new(BTreeMap::new())),
            current_block: Arc::new(RwLock::new(0)),
        }
    }

    /// Submit an architecture proposal
    pub async fn submit_proposal(
        &self,
        proposer: [u8; 32],
        architecture: Architecture,
        description: String,
        parent_id: Option<u64>,
    ) -> anyhow::Result<u64> {
        let stakes = self.stakes.read().await;
        let proposer_stake = stakes.get(&proposer).copied().unwrap_or(0);

        if proposer_stake < self.config.min_proposal_stake {
            return Err(anyhow::anyhow!(
                "Insufficient stake: {} < {}",
                proposer_stake,
                self.config.min_proposal_stake
            ));
        }

        let current_block = *self.current_block.read().await;
        let proposals = self.proposals.read().await;

        // Check concurrent proposal limit
        let active_count = proposals.values()
            .filter(|p| matches!(p.status, ProposalStatus::Pending | ProposalStatus::Voting { .. }))
            .count();

        if active_count >= self.config.max_concurrent_proposals {
            return Err(anyhow::anyhow!("Too many active proposals"));
        }

        // Compute architecture hash for deduplication
        let architecture_hash = self.compute_architecture_hash(&architecture);

        // Check for duplicates
        for existing in proposals.values() {
            if existing.architecture_hash == architecture_hash {
                return Err(anyhow::anyhow!("Duplicate architecture already proposed"));
            }
        }
        drop(proposals);

        // Evaluate preliminary fitness
        let preliminary_fitness = {
            let nas = self.nas_engine.read().await;
            Some(nas.evaluate_fitness(&architecture))
        };

        // Create proposal
        let mut state = self.state.write().await;
        let proposal_id = state.next_proposal_id;
        state.next_proposal_id += 1;
        state.total_proposals += 1;
        drop(state);

        let proposal = ArchitectureProposal {
            id: proposal_id,
            proposer,
            architecture,
            bond: self.config.proposal_bond,
            submitted_at: current_block,
            status: ProposalStatus::Pending,
            preliminary_fitness,
            on_chain_metrics: None,
            description,
            parent_id,
            architecture_hash,
        };

        self.proposals.write().await.insert(proposal_id, proposal);
        self.votes.write().await.insert(proposal_id, Vec::new());

        info!("📜 Proposal {} submitted by {:?}",
              proposal_id, hex::encode(&proposer[..8]));

        Ok(proposal_id)
    }

    /// Start voting on a proposal
    pub async fn start_voting(&self, proposal_id: u64) -> anyhow::Result<()> {
        let mut proposals = self.proposals.write().await;
        let proposal = proposals.get_mut(&proposal_id)
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        if proposal.status != ProposalStatus::Pending {
            return Err(anyhow::anyhow!("Proposal not in pending state"));
        }

        let current_block = *self.current_block.read().await;
        let end_block = current_block + self.config.voting_period_blocks;

        proposal.status = ProposalStatus::Voting { end_block };

        info!("🗳️ Voting started on proposal {} (ends at block {})",
              proposal_id, end_block);

        Ok(())
    }

    /// Cast a vote on a proposal
    pub async fn vote(
        &self,
        voter: [u8; 32],
        proposal_id: u64,
        direction: VoteDirection,
        rationale: Option<String>,
    ) -> anyhow::Result<()> {
        let proposals = self.proposals.read().await;
        let proposal = proposals.get(&proposal_id)
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        // Verify voting period
        match &proposal.status {
            ProposalStatus::Voting { end_block } => {
                let current = *self.current_block.read().await;
                if current > *end_block {
                    return Err(anyhow::anyhow!("Voting period ended"));
                }
            }
            _ => return Err(anyhow::anyhow!("Proposal not in voting phase")),
        }
        drop(proposals);

        // Check if already voted
        let votes = self.votes.read().await;
        if let Some(existing_votes) = votes.get(&proposal_id) {
            if existing_votes.iter().any(|v| v.voter == voter) {
                return Err(anyhow::anyhow!("Already voted on this proposal"));
            }
        }
        drop(votes);

        // Calculate voting power
        let stakes = self.stakes.read().await;
        let stake = stakes.get(&voter).copied().unwrap_or(0);
        if stake == 0 {
            return Err(anyhow::anyhow!("No stake to vote with"));
        }

        let voting_power = if self.config.quadratic_voting {
            (stake as f64).sqrt() as u64
        } else {
            stake
        };

        let vote = Vote {
            voter,
            proposal_id,
            direction,
            voting_power,
            timestamp: chrono::Utc::now().timestamp() as u64,
            rationale,
        };

        self.votes.write().await
            .entry(proposal_id)
            .or_default()
            .push(vote);

        let mut state = self.state.write().await;
        state.total_votes += 1;

        info!("🗳️ Vote cast on proposal {} by {:?}",
              proposal_id, hex::encode(&voter[..8]));

        Ok(())
    }

    /// Tally votes and finalize proposal
    pub async fn finalize_proposal(&self, proposal_id: u64) -> anyhow::Result<ProposalStatus> {
        let mut proposals = self.proposals.write().await;
        let proposal = proposals.get_mut(&proposal_id)
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        // Verify voting period ended
        let current_block = *self.current_block.read().await;
        match &proposal.status {
            ProposalStatus::Voting { end_block } => {
                if current_block < *end_block {
                    return Err(anyhow::anyhow!("Voting period not ended"));
                }
            }
            _ => return Err(anyhow::anyhow!("Proposal not in voting phase")),
        }

        // Tally votes
        let votes = self.votes.read().await;
        let proposal_votes = votes.get(&proposal_id).cloned().unwrap_or_default();
        drop(votes);

        let mut for_power = 0u64;
        let mut against_power = 0u64;
        let mut abstain_power = 0u64;

        for vote in &proposal_votes {
            match vote.direction {
                VoteDirection::For => for_power += vote.voting_power,
                VoteDirection::Against => against_power += vote.voting_power,
                VoteDirection::Abstain => abstain_power += vote.voting_power,
            }
        }

        let total_power = for_power + against_power + abstain_power;

        // Calculate total stake for quorum
        let stakes = self.stakes.read().await;
        let total_stake: u64 = stakes.values().sum();
        let total_voting_stake = if self.config.quadratic_voting {
            (total_stake as f64).sqrt() as u64
        } else {
            total_stake
        };
        drop(stakes);

        // Check quorum
        let participation = if total_voting_stake > 0 {
            (total_power * 100) / total_voting_stake
        } else {
            0
        };

        if participation < self.config.quorum_percentage as u64 {
            proposal.status = ProposalStatus::Rejected;
            info!("❌ Proposal {} rejected: quorum not met ({}/{}%)",
                  proposal_id, participation, self.config.quorum_percentage);
            return Ok(ProposalStatus::Rejected);
        }

        // Check approval threshold
        let approval = if for_power + against_power > 0 {
            (for_power * 100) / (for_power + against_power)
        } else {
            0
        };

        if approval >= self.config.approval_threshold as u64 {
            let execute_at = current_block + self.config.execution_delay_blocks;
            proposal.status = ProposalStatus::Approved { execute_at };
            info!("✅ Proposal {} approved with {}% (execute at block {})",
                  proposal_id, approval, execute_at);
            Ok(ProposalStatus::Approved { execute_at })
        } else {
            proposal.status = ProposalStatus::Rejected;
            info!("❌ Proposal {} rejected with {}% approval",
                  proposal_id, approval);
            Ok(ProposalStatus::Rejected)
        }
    }

    /// Execute an approved proposal
    pub async fn execute_proposal(&self, proposal_id: u64) -> anyhow::Result<()> {
        let mut proposals = self.proposals.write().await;
        let proposal = proposals.get_mut(&proposal_id)
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        let current_block = *self.current_block.read().await;

        match &proposal.status {
            ProposalStatus::Approved { execute_at } => {
                if current_block < *execute_at {
                    return Err(anyhow::anyhow!(
                        "Execution delay not passed: {} < {}",
                        current_block, execute_at
                    ));
                }
            }
            _ => return Err(anyhow::anyhow!("Proposal not approved")),
        }

        // Update active architecture
        let mut state = self.state.write().await;

        // Record retirement of previous architecture
        if let Some(prev_id) = state.active_id {
            let mut history = self.history.write().await;
            if let Some(perf) = history.get_mut(&prev_id) {
                perf.retirement_block = Some(current_block);
            }
        }

        state.active_architecture = Some(proposal.architecture.clone());
        state.active_id = Some(proposal_id);
        proposal.status = ProposalStatus::Deployed { deployed_at: current_block };

        // Record in history
        let fitness = proposal.preliminary_fitness.clone().unwrap_or_else(|| {
            FitnessScore {
                accuracy: 0.0,
                efficiency: 0.0,
                latency: 0.0,
                complexity: 0.0,
            }
        });

        self.history.write().await.insert(proposal_id, ArchitecturePerformance {
            architecture_id: proposal_id,
            architecture_hash: proposal.architecture_hash,
            deployment_block: current_block,
            retirement_block: None,
            metrics: OnChainMetrics {
                accuracy: 0.0,
                avg_latency_ms: 0.0,
                total_predictions: 0,
                successful_predictions: 0,
                revenue_generated: 0,
                compute_cost: 0,
                start_block: current_block,
                end_block: 0,
            },
            fitness,
        });

        info!("🚀 Architecture {} deployed at block {}",
              proposal_id, current_block);

        Ok(())
    }

    /// Generate evolution proposals using NAS engine
    pub async fn generate_evolution_proposals(
        &self,
        num_proposals: usize,
    ) -> anyhow::Result<Vec<Architecture>> {
        let state = self.state.read().await;

        let current_arch = state.active_architecture.clone()
            .ok_or_else(|| anyhow::anyhow!("No active architecture to evolve from"))?;

        let mut nas = self.nas_engine.write().await;

        // Seed population with current architecture
        nas.seed_population(vec![current_arch.clone()]);

        // Run evolution for one generation
        nas.evolve_generation();

        // Get top candidates
        let population = nas.get_population();
        let top: Vec<Architecture> = population.iter()
            .take(num_proposals)
            .cloned()
            .collect();

        info!("🧬 Generated {} evolution proposals", top.len());

        Ok(top)
    }

    /// Auto-approve proposals that exceed fitness threshold
    pub async fn check_auto_approval(&self, proposal_id: u64) -> anyhow::Result<bool> {
        let proposals = self.proposals.read().await;
        let proposal = proposals.get(&proposal_id)
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        let state = self.state.read().await;

        // Compare with current architecture fitness
        if let (Some(proposed_fitness), Some(current_arch)) = (
            &proposal.preliminary_fitness,
            &state.active_architecture
        ) {
            let nas = self.nas_engine.read().await;
            let current_fitness = nas.evaluate_fitness(current_arch);

            let improvement = (proposed_fitness.accuracy - current_fitness.accuracy)
                            / current_fitness.accuracy.max(0.001);

            if improvement > self.config.auto_approve_fitness_delta {
                info!("⚡ Proposal {} qualifies for auto-approval ({:.1}% improvement)",
                      proposal_id, improvement * 100.0);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Update on-chain metrics for active architecture
    pub async fn update_metrics(&self, metrics: OnChainMetrics) -> anyhow::Result<()> {
        let state = self.state.read().await;

        if let Some(arch_id) = state.active_id {
            let mut history = self.history.write().await;
            if let Some(perf) = history.get_mut(&arch_id) {
                perf.metrics = metrics;
            }
        }

        Ok(())
    }

    /// Compute architecture hash
    fn compute_architecture_hash(&self, arch: &Architecture) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"arch_v1");

        for layer in &arch.layers {
            hasher.update(&[layer.layer_type as u8]);
            hasher.update(&(layer.input_dim as u64).to_le_bytes());
            hasher.update(&(layer.output_dim as u64).to_le_bytes());
        }

        hasher.finalize().into()
    }

    /// Set validator stake
    pub async fn set_stake(&self, validator: [u8; 32], stake: u64) {
        self.stakes.write().await.insert(validator, stake);
    }

    /// Advance block height
    pub async fn advance_block(&self, height: u64) {
        *self.current_block.write().await = height;
    }

    /// Get current active architecture
    pub async fn get_active_architecture(&self) -> Option<Architecture> {
        self.state.read().await.active_architecture.clone()
    }

    /// Get proposal by ID
    pub async fn get_proposal(&self, id: u64) -> Option<ArchitectureProposal> {
        self.proposals.read().await.get(&id).cloned()
    }

    /// Get all active proposals
    pub async fn get_active_proposals(&self) -> Vec<ArchitectureProposal> {
        self.proposals.read().await
            .values()
            .filter(|p| matches!(
                p.status,
                ProposalStatus::Pending | ProposalStatus::Voting { .. }
            ))
            .cloned()
            .collect()
    }

    /// Get governance statistics
    pub async fn get_stats(&self) -> GovernanceStats {
        let state = self.state.read().await;
        let proposals = self.proposals.read().await;
        let history = self.history.read().await;

        let deployed_count = proposals.values()
            .filter(|p| matches!(p.status, ProposalStatus::Deployed { .. }))
            .count();

        let rejected_count = proposals.values()
            .filter(|p| matches!(p.status, ProposalStatus::Rejected))
            .count();

        GovernanceStats {
            total_proposals: state.total_proposals,
            total_votes: state.total_votes,
            deployed_architectures: deployed_count as u64,
            rejected_proposals: rejected_count as u64,
            current_architecture_id: state.active_id,
            architecture_changes: history.len() as u64,
        }
    }
}

/// Governance statistics
#[derive(Clone, Debug)]
pub struct GovernanceStats {
    pub total_proposals: u64,
    pub total_votes: u64,
    pub deployed_architectures: u64,
    pub rejected_proposals: u64,
    pub current_architecture_id: Option<u64>,
    pub architecture_changes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::LayerGene;

    fn create_test_architecture() -> Architecture {
        Architecture {
            id: 0,
            layers: vec![
                LayerGene {
                    layer_type: LayerType::Dense,
                    input_dim: 64,
                    output_dim: 32,
                    activation: Some("relu".to_string()),
                    dropout: Some(0.1),
                    batch_norm: true,
                },
                LayerGene {
                    layer_type: LayerType::Dense,
                    input_dim: 32,
                    output_dim: 10,
                    activation: Some("softmax".to_string()),
                    dropout: None,
                    batch_norm: false,
                },
            ],
            fitness: None,
            pareto_rank: 0,
            crowding_distance: 0.0,
        }
    }

    #[tokio::test]
    async fn test_governance_creation() {
        let config = GovernanceConfig::default();
        let gov = NasGovernance::new(config);

        let stats = gov.get_stats().await;
        assert_eq!(stats.total_proposals, 0);
    }

    #[tokio::test]
    async fn test_proposal_submission() {
        let config = GovernanceConfig {
            min_proposal_stake: 100,
            ..Default::default()
        };
        let gov = NasGovernance::new(config);

        let proposer = [1u8; 32];
        gov.set_stake(proposer, 1000).await;

        let arch = create_test_architecture();
        let proposal_id = gov.submit_proposal(
            proposer,
            arch,
            "Test proposal".to_string(),
            None,
        ).await.unwrap();

        assert_eq!(proposal_id, 1);

        let proposal = gov.get_proposal(proposal_id).await.unwrap();
        assert_eq!(proposal.status, ProposalStatus::Pending);
    }

    #[tokio::test]
    async fn test_voting_flow() {
        let config = GovernanceConfig {
            min_proposal_stake: 100,
            voting_period_blocks: 10,
            quorum_percentage: 10,
            approval_threshold: 50,
            ..Default::default()
        };
        let gov = NasGovernance::new(config);

        let proposer = [1u8; 32];
        let voter1 = [2u8; 32];
        let voter2 = [3u8; 32];

        gov.set_stake(proposer, 1000).await;
        gov.set_stake(voter1, 500).await;
        gov.set_stake(voter2, 500).await;

        // Submit proposal
        let arch = create_test_architecture();
        let proposal_id = gov.submit_proposal(
            proposer,
            arch,
            "Test".to_string(),
            None,
        ).await.unwrap();

        // Start voting
        gov.start_voting(proposal_id).await.unwrap();

        // Cast votes
        gov.vote(voter1, proposal_id, VoteDirection::For, None).await.unwrap();
        gov.vote(voter2, proposal_id, VoteDirection::For, None).await.unwrap();

        // Advance past voting period
        gov.advance_block(15).await;

        // Finalize
        let status = gov.finalize_proposal(proposal_id).await.unwrap();
        assert!(matches!(status, ProposalStatus::Approved { .. }));
    }

    #[tokio::test]
    async fn test_architecture_hash() {
        let gov = NasGovernance::new(GovernanceConfig::default());

        let arch1 = create_test_architecture();
        let arch2 = create_test_architecture();

        let hash1 = gov.compute_architecture_hash(&arch1);
        let hash2 = gov.compute_architecture_hash(&arch2);

        assert_eq!(hash1, hash2); // Same architecture = same hash
    }
}
