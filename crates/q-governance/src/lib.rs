//! Q-Governance: Proof-of-Contribution Governance System
//!
//! Implements mining-weighted voting without the security flaws of mining-enhanced signatures.
//! Uses hashpower contribution to weight governance votes in a sybil-resistant manner.
//!
//! ## Architecture
//!
//! ```text
//! Voting Power = TokenStake × (1 + log₂(HashesContributed) / 100)
//! ```
//!
//! This provides:
//! - **Sybil Resistance**: Requires computational work to amplify voting power
//! - **Economic Rationality**: Miners rewarded with governance influence
//! - **Logarithmic Scaling**: Prevents whale attacks (diminishing returns)
//! - **No Security Composition**: Doesn't claim to strengthen cryptographic signatures
//!
//! ## v2.4.0-beta: Persistence
//!
//! Proposals and votes are now persisted to RocksDB via StorageEngine.
//! On restart, all governance data is restored automatically.

use anyhow::{anyhow, Result};
use q_storage::StorageEngine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub mod types;
pub mod voting;
pub mod mining_contribution;
pub mod reputation;

pub use types::*;
pub use voting::*;
pub use mining_contribution::*;
pub use reputation::*;

/// Governance coordinator managing proposals and votes
/// v2.4.0-beta: Now persists to RocksDB for crash resilience
pub struct GovernanceCoordinator {
    /// Active proposals (in-memory cache backed by RocksDB)
    proposals: Arc<RwLock<HashMap<String, Proposal>>>,

    /// Vote registry (in-memory cache backed by RocksDB)
    votes: Arc<RwLock<HashMap<String, Vec<WeightedVote>>>>,

    /// Mining contribution tracker
    contribution_tracker: Arc<MiningContributionTracker>,

    /// Miner reputation system
    reputation_system: Arc<ReputationSystem>,

    /// Voting power calculator
    power_calculator: VotingPowerCalculator,

    /// v2.4.0-beta: Storage engine for persistence
    storage: Option<Arc<StorageEngine>>,
}

impl GovernanceCoordinator {
    /// Create new governance coordinator (in-memory only, for testing)
    pub fn new() -> Self {
        Self {
            proposals: Arc::new(RwLock::new(HashMap::new())),
            votes: Arc::new(RwLock::new(HashMap::new())),
            contribution_tracker: Arc::new(MiningContributionTracker::new()),
            reputation_system: Arc::new(ReputationSystem::new()),
            power_calculator: VotingPowerCalculator::new(),
            storage: None,
        }
    }

    /// Create governance coordinator with RocksDB persistence
    /// v2.4.0-beta: Loads existing proposals and votes from storage on startup
    pub async fn with_storage(storage: Arc<StorageEngine>) -> Self {
        let mut proposals_map = HashMap::new();
        let mut votes_map: HashMap<String, Vec<WeightedVote>> = HashMap::new();

        // Load existing proposals from RocksDB
        match storage.load_all_governance_proposals().await {
            Ok(stored_proposals) => {
                for (proposal_id, bytes) in stored_proposals {
                    match bincode::deserialize::<Proposal>(&bytes) {
                        Ok(proposal) => {
                            info!("📜 Restored governance proposal: {}", proposal_id);
                            proposals_map.insert(proposal_id.clone(), proposal);

                            // Load votes for this proposal
                            match storage.load_governance_votes_for_proposal(&proposal_id).await {
                                Ok(vote_bytes_list) => {
                                    let mut proposal_votes = Vec::new();
                                    for vote_bytes in vote_bytes_list {
                                        if let Ok(vote) = bincode::deserialize::<WeightedVote>(&vote_bytes) {
                                            proposal_votes.push(vote);
                                        }
                                    }
                                    if !proposal_votes.is_empty() {
                                        info!("📜 Restored {} votes for proposal {}", proposal_votes.len(), proposal_id);
                                        votes_map.insert(proposal_id, proposal_votes);
                                    }
                                }
                                Err(e) => {
                                    warn!("⚠️ Failed to load votes for proposal {}: {}", proposal_id, e);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("⚠️ Failed to deserialize proposal {}: {}", proposal_id, e);
                        }
                    }
                }
                if !proposals_map.is_empty() {
                    info!("✅ Restored {} governance proposals from RocksDB", proposals_map.len());
                }
            }
            Err(e) => {
                warn!("⚠️ Failed to load governance proposals: {}", e);
            }
        }

        Self {
            proposals: Arc::new(RwLock::new(proposals_map)),
            votes: Arc::new(RwLock::new(votes_map)),
            contribution_tracker: Arc::new(MiningContributionTracker::new()),
            reputation_system: Arc::new(ReputationSystem::new()),
            power_calculator: VotingPowerCalculator::new(),
            storage: Some(storage),
        }
    }

    /// Create new governance proposal
    /// v2.4.0-beta: Now persists to RocksDB for crash resilience
    pub async fn create_proposal(
        &self,
        proposal: Proposal,
    ) -> Result<String> {
        let proposal_id = proposal.id.clone();

        // Validate proposal
        self.validate_proposal(&proposal)?;

        // Persist to RocksDB first (if storage available)
        if let Some(ref storage) = self.storage {
            let proposal_bytes = bincode::serialize(&proposal)
                .map_err(|e| anyhow!("Failed to serialize proposal: {}", e))?;
            storage.save_governance_proposal(&proposal_id, &proposal_bytes).await
                .map_err(|e| anyhow!("Failed to persist proposal: {}", e))?;
            debug!("💾 Persisted governance proposal to RocksDB: {}", proposal_id);
        }

        // Store proposal in memory
        let mut proposals = self.proposals.write().await;
        proposals.insert(proposal_id.clone(), proposal);

        info!("✅ Created governance proposal: {}", proposal_id);

        Ok(proposal_id)
    }

    /// Submit weighted vote with optional mining contribution
    /// v2.4.0-beta: Now persists to RocksDB for crash resilience
    pub async fn submit_vote(
        &self,
        vote: WeightedVote,
    ) -> Result<()> {
        // Validate vote
        self.validate_vote(&vote).await?;

        // Verify mining contribution if provided
        if let Some(ref contribution) = vote.mining_contribution {
            self.contribution_tracker
                .verify_contribution(contribution)
                .await?;
        }

        // Calculate final voting power
        let voting_power = self.power_calculator.calculate_power(
            vote.token_stake,
            vote.mining_contribution.as_ref(),
        );

        // Persist to RocksDB first (if storage available)
        if let Some(ref storage) = self.storage {
            let vote_bytes = bincode::serialize(&vote)
                .map_err(|e| anyhow!("Failed to serialize vote: {}", e))?;
            let voter_hex = hex::encode(&vote.voter_address);
            storage.save_governance_vote(&vote.proposal_id, &voter_hex, &vote_bytes).await
                .map_err(|e| anyhow!("Failed to persist vote: {}", e))?;
            debug!("💾 Persisted governance vote to RocksDB: {} from {}", vote.proposal_id, voter_hex);
        }

        // Store vote in memory
        let mut votes = self.votes.write().await;
        votes
            .entry(vote.proposal_id.clone())
            .or_insert_with(Vec::new)
            .push(vote.clone());

        info!("✅ Vote submitted: proposal={}, power={}",
              vote.proposal_id, voting_power);

        Ok(())
    }

    /// Calculate proposal results
    pub async fn calculate_results(
        &self,
        proposal_id: &str,
    ) -> Result<ProposalResult> {
        let votes = self.votes.read().await;

        let proposal_votes = votes
            .get(proposal_id)
            .ok_or_else(|| anyhow!("Proposal not found: {}", proposal_id))?;

        let mut total_power = 0u128;
        let mut option_powers: HashMap<String, u128> = HashMap::new();

        for vote in proposal_votes {
            let power = self.power_calculator.calculate_power(
                vote.token_stake,
                vote.mining_contribution.as_ref(),
            );

            total_power += power;
            *option_powers.entry(vote.vote_choice.clone()).or_insert(0) += power;
        }

        // Find winning option
        let winning_option = option_powers
            .iter()
            .max_by_key(|(_, power)| *power)
            .map(|(option, power)| (option.clone(), *power));

        Ok(ProposalResult {
            proposal_id: proposal_id.to_string(),
            total_votes: proposal_votes.len(),
            total_voting_power: total_power,
            option_results: option_powers,
            winning_option,
            finalized: true,
        })
    }

    /// Validate proposal
    fn validate_proposal(&self, proposal: &Proposal) -> Result<()> {
        if proposal.title.is_empty() {
            return Err(anyhow!("Proposal title cannot be empty"));
        }

        if proposal.options.len() < 2 {
            return Err(anyhow!("Proposal must have at least 2 options"));
        }

        if proposal.voting_end <= proposal.voting_start {
            return Err(anyhow!("Voting end must be after voting start"));
        }

        Ok(())
    }

    /// Validate vote
    async fn validate_vote(&self, vote: &WeightedVote) -> Result<()> {
        // Check proposal exists
        let proposals = self.proposals.read().await;
        let proposal = proposals
            .get(&vote.proposal_id)
            .ok_or_else(|| anyhow!("Proposal not found: {}", vote.proposal_id))?;

        // Check voting period
        let now = chrono::Utc::now().timestamp() as u64;
        if now < proposal.voting_start {
            return Err(anyhow!("Voting has not started yet"));
        }
        if now > proposal.voting_end {
            return Err(anyhow!("Voting has ended"));
        }

        // Check vote choice is valid
        if !proposal.options.contains(&vote.vote_choice) {
            return Err(anyhow!("Invalid vote choice: {}", vote.vote_choice));
        }

        // Check token stake is non-zero
        if vote.token_stake == 0 {
            return Err(anyhow!("Token stake must be greater than zero"));
        }

        Ok(())
    }

    /// Get proposal details
    pub async fn get_proposal(&self, proposal_id: &str) -> Result<Proposal> {
        let proposals = self.proposals.read().await;
        proposals
            .get(proposal_id)
            .cloned()
            .ok_or_else(|| anyhow!("Proposal not found: {}", proposal_id))
    }

    /// Get all active proposals
    pub async fn get_active_proposals(&self) -> Vec<Proposal> {
        let proposals = self.proposals.read().await;
        let now = chrono::Utc::now().timestamp() as u64;

        proposals
            .values()
            .filter(|p| p.voting_start <= now && now <= p.voting_end)
            .cloned()
            .collect()
    }

    /// Get all proposals (active and inactive)
    pub async fn get_all_proposals(&self) -> Vec<Proposal> {
        let proposals = self.proposals.read().await;
        proposals.values().cloned().collect()
    }

    /// Calculate voting power for a given vote
    pub fn calculate_voting_power(&self, vote: &WeightedVote) -> u128 {
        self.power_calculator.calculate_power(
            vote.token_stake,
            vote.mining_contribution.as_ref(),
        )
    }

    /// Get contribution stats for an address (delegate to contribution_tracker)
    pub async fn get_contribution_stats(&self, address: &[u8; 32]) -> ContributionStats {
        self.contribution_tracker.get_contribution_stats(address).await
    }

    /// Get reputation multiplier for an address (delegate to reputation_system)
    pub async fn get_reputation_multiplier(&self, address: &[u8; 32]) -> f64 {
        self.reputation_system
            .get_reputation(address)
            .await
            .map(|rep| rep.reputation_multiplier)
            .unwrap_or(1.0)
    }

    /// Get reputation for an address (delegate to reputation_system)
    pub async fn get_reputation(&self, address: &[u8; 32]) -> Option<MinerReputation> {
        self.reputation_system.get_reputation(address).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_proposal() {
        let coordinator = GovernanceCoordinator::new();

        let proposal = Proposal {
            id: "test-proposal-1".to_string(),
            title: "Test Proposal".to_string(),
            description: "This is a test proposal".to_string(),
            proposer: [1u8; 32],
            options: vec!["Yes".to_string(), "No".to_string()],
            voting_start: chrono::Utc::now().timestamp() as u64,
            voting_end: chrono::Utc::now().timestamp() as u64 + 86400,
            proposal_type: ProposalType::ParameterChange,
            required_quorum: 1000,
        };

        let result = coordinator.create_proposal(proposal).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_voting_power_calculation() {
        let calculator = VotingPowerCalculator::new();

        // No mining contribution
        let power1 = calculator.calculate_power(1000, None);
        assert_eq!(power1, 1000);

        // With mining contribution (1 million hashes)
        let contribution = MiningContribution {
            solutions: vec![],
            total_hashes: 1_000_000,
            merkle_proofs: vec![],
            contribution_period: (0, 1000),
        };

        let power2 = calculator.calculate_power(1000, Some(&contribution));
        assert!(power2 > 1000); // Should be higher with mining
        assert!(power2 < 1200); // But not too much higher (logarithmic)
    }
}
