// Security Tier Governance System
// Community-based voting for adaptive VDF security parameters
//
// Design: Addresses AI feedback about perverse economic incentives
// Instead of automatic hashrate scaling, security tiers are voted by the community

use anyhow::{anyhow, Result};
use q_types::block::{AdaptiveVDFParams, SecurityTier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Security tier governance proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityTierProposal {
    pub proposal_id: u64,
    pub proposed_tier: SecurityTier,
    pub proposer: String, // Wallet address
    pub created_at: u64,
    pub voting_ends_at: u64,
    pub description: String,
    pub rationale: String, // Why this tier is needed
    pub votes: HashMap<String, SecurityTierVote>,
    pub status: ProposalStatus,
}

/// Individual vote on a security tier proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityTierVote {
    pub voter: String, // Wallet address
    pub vote: VoteChoice,
    pub voting_power: u64, // Weighted by mining contribution
    pub timestamp: u64,
}

/// Vote choice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

/// Proposal status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Active,
    Passed,
    Rejected,
    Expired,
}

/// Security tier governance manager
pub struct SecurityTierGovernance {
    /// Active proposals
    proposals: Arc<RwLock<HashMap<u64, SecurityTierProposal>>>,

    /// Current active security tier
    current_tier: Arc<RwLock<SecurityTier>>,

    /// Mining contribution tracking (for voting power)
    mining_contributions: Arc<RwLock<HashMap<String, u64>>>,

    /// Voting period duration (7 days)
    voting_period: Duration,

    /// Minimum quorum (30% of total mining power)
    min_quorum: f64,

    /// Approval threshold (66% supermajority)
    approval_threshold: f64,

    /// Next proposal ID
    next_proposal_id: Arc<RwLock<u64>>,
}

impl SecurityTierGovernance {
    pub fn new() -> Self {
        Self {
            proposals: Arc::new(RwLock::new(HashMap::new())),
            current_tier: Arc::new(RwLock::new(SecurityTier::Standard)),
            mining_contributions: Arc::new(RwLock::new(HashMap::new())),
            voting_period: Duration::from_secs(7 * 86400), // 7 days
            min_quorum: 0.30,                              // 30% quorum
            approval_threshold: 0.66,                      // 66% approval
            next_proposal_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Create a new security tier proposal
    pub async fn create_proposal(
        &self,
        proposer: String,
        proposed_tier: SecurityTier,
        description: String,
        rationale: String,
    ) -> Result<u64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let proposal_id = {
            let mut next_id = self.next_proposal_id.write().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let voting_ends_at = now + self.voting_period.as_secs();

        let proposal = SecurityTierProposal {
            proposal_id,
            proposed_tier,
            proposer: proposer.clone(),
            created_at: now,
            voting_ends_at,
            description,
            rationale,
            votes: HashMap::new(),
            status: ProposalStatus::Active,
        };

        self.proposals.write().await.insert(proposal_id, proposal);

        info!(
            "Security tier proposal {} created by {} for {:?}",
            proposal_id, proposer, proposed_tier
        );

        Ok(proposal_id)
    }

    /// Cast a vote on a proposal
    pub async fn vote(&self, proposal_id: u64, voter: String, vote: VoteChoice) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get voting power based on mining contributions
        let voting_power = {
            let contributions = self.mining_contributions.read().await;
            *contributions.get(&voter).unwrap_or(&0)
        };

        if voting_power == 0 {
            return Err(anyhow!(
                "Voter {} has no mining contributions (zero voting power)",
                voter
            ));
        }

        // Update proposal with vote
        let mut proposals = self.proposals.write().await;
        let proposal = proposals
            .get_mut(&proposal_id)
            .ok_or_else(|| anyhow!("Proposal {} not found", proposal_id))?;

        if proposal.status != ProposalStatus::Active {
            return Err(anyhow!("Proposal {} is not active", proposal_id));
        }

        if now > proposal.voting_ends_at {
            proposal.status = ProposalStatus::Expired;
            return Err(anyhow!("Proposal {} voting period has ended", proposal_id));
        }

        let vote_record = SecurityTierVote {
            voter: voter.clone(),
            vote,
            voting_power,
            timestamp: now,
        };

        proposal.votes.insert(voter.clone(), vote_record);

        info!(
            "Vote recorded: {} voted {:?} on proposal {} with {} power",
            voter, vote, proposal_id, voting_power
        );

        Ok(())
    }

    /// Finalize a proposal after voting period ends
    pub async fn finalize_proposal(&self, proposal_id: u64) -> Result<SecurityTier> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut proposals = self.proposals.write().await;
        let proposal = proposals
            .get_mut(&proposal_id)
            .ok_or_else(|| anyhow!("Proposal {} not found", proposal_id))?;

        if proposal.status != ProposalStatus::Active {
            return Err(anyhow!("Proposal {} is not active", proposal_id));
        }

        if now < proposal.voting_ends_at {
            return Err(anyhow!(
                "Proposal {} voting period has not ended yet",
                proposal_id
            ));
        }

        // Calculate vote totals
        let total_mining_power: u64 = self.mining_contributions.read().await.values().sum();
        let mut votes_for = 0u64;
        let mut votes_against = 0u64;
        let mut votes_abstain = 0u64;

        for vote in proposal.votes.values() {
            match vote.vote {
                VoteChoice::For => votes_for += vote.voting_power,
                VoteChoice::Against => votes_against += vote.voting_power,
                VoteChoice::Abstain => votes_abstain += vote.voting_power,
            }
        }

        let total_votes = votes_for + votes_against + votes_abstain;
        let participation = total_votes as f64 / total_mining_power as f64;
        let approval_rate = votes_for as f64 / (votes_for + votes_against) as f64;

        // Check quorum
        if participation < self.min_quorum {
            proposal.status = ProposalStatus::Rejected;
            warn!(
                "Proposal {} rejected: quorum not met ({:.1}% < {:.1}%)",
                proposal_id,
                participation * 100.0,
                self.min_quorum * 100.0
            );
            return Err(anyhow!("Quorum not met"));
        }

        // Check approval threshold
        if approval_rate >= self.approval_threshold {
            proposal.status = ProposalStatus::Passed;

            // Update current tier
            *self.current_tier.write().await = proposal.proposed_tier;

            info!(
                "Proposal {} PASSED: {:?} tier activated ({:.1}% approval)",
                proposal_id,
                proposal.proposed_tier,
                approval_rate * 100.0
            );

            Ok(proposal.proposed_tier)
        } else {
            proposal.status = ProposalStatus::Rejected;
            warn!(
                "Proposal {} REJECTED: approval threshold not met ({:.1}% < {:.1}%)",
                proposal_id,
                approval_rate * 100.0,
                self.approval_threshold * 100.0
            );
            Err(anyhow!("Approval threshold not met"))
        }
    }

    /// Record mining contribution (updates voting power)
    pub async fn record_mining_contribution(&self, miner: String, contribution: u64) -> Result<()> {
        let mut contributions = self.mining_contributions.write().await;
        *contributions.entry(miner.clone()).or_insert(0) += contribution;

        Ok(())
    }

    /// Get current active security tier
    pub async fn get_current_tier(&self) -> SecurityTier {
        *self.current_tier.read().await
    }

    /// Get proposal details
    pub async fn get_proposal(&self, proposal_id: u64) -> Option<SecurityTierProposal> {
        self.proposals.read().await.get(&proposal_id).cloned()
    }

    /// Get all active proposals
    pub async fn get_active_proposals(&self) -> Vec<SecurityTierProposal> {
        self.proposals
            .read()
            .await
            .values()
            .filter(|p| p.status == ProposalStatus::Active)
            .cloned()
            .collect()
    }

    /// Get voting statistics for a proposal
    pub async fn get_voting_stats(&self, proposal_id: u64) -> Result<VotingStatistics> {
        let proposals = self.proposals.read().await;
        let proposal = proposals
            .get(&proposal_id)
            .ok_or_else(|| anyhow!("Proposal {} not found", proposal_id))?;

        let total_mining_power: u64 = self.mining_contributions.read().await.values().sum();
        let mut votes_for = 0u64;
        let mut votes_against = 0u64;
        let mut votes_abstain = 0u64;

        for vote in proposal.votes.values() {
            match vote.vote {
                VoteChoice::For => votes_for += vote.voting_power,
                VoteChoice::Against => votes_against += vote.voting_power,
                VoteChoice::Abstain => votes_abstain += vote.voting_power,
            }
        }

        let total_votes = votes_for + votes_against + votes_abstain;
        let participation = total_votes as f64 / total_mining_power as f64;
        let approval_rate = if votes_for + votes_against > 0 {
            votes_for as f64 / (votes_for + votes_against) as f64
        } else {
            0.0
        };

        Ok(VotingStatistics {
            proposal_id,
            votes_for,
            votes_against,
            votes_abstain,
            total_votes,
            total_mining_power,
            participation_rate: participation,
            approval_rate,
            quorum_met: participation >= self.min_quorum,
            will_pass: approval_rate >= self.approval_threshold && participation >= self.min_quorum,
        })
    }
}

/// Voting statistics for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingStatistics {
    pub proposal_id: u64,
    pub votes_for: u64,
    pub votes_against: u64,
    pub votes_abstain: u64,
    pub total_votes: u64,
    pub total_mining_power: u64,
    pub participation_rate: f64,
    pub approval_rate: f64,
    pub quorum_met: bool,
    pub will_pass: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_proposal() {
        let governance = SecurityTierGovernance::new();

        let proposal_id = governance
            .create_proposal(
                "founder".to_string(),
                SecurityTier::Enhanced,
                "Upgrade to Enhanced tier".to_string(),
                "Network hashrate has grown 10x".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(proposal_id, 1);
    }

    #[tokio::test]
    async fn test_voting() {
        let governance = SecurityTierGovernance::new();

        // Record mining contributions
        governance
            .record_mining_contribution("alice".to_string(), 1000)
            .await
            .unwrap();
        governance
            .record_mining_contribution("bob".to_string(), 500)
            .await
            .unwrap();

        // Create proposal
        let proposal_id = governance
            .create_proposal(
                "alice".to_string(),
                SecurityTier::Maximum,
                "Upgrade to Maximum tier".to_string(),
                "Critical security needed".to_string(),
            )
            .await
            .unwrap();

        // Vote
        governance
            .vote(proposal_id, "alice".to_string(), VoteChoice::For)
            .await
            .unwrap();
        governance
            .vote(proposal_id, "bob".to_string(), VoteChoice::For)
            .await
            .unwrap();

        // Get stats
        let stats = governance.get_voting_stats(proposal_id).await.unwrap();
        assert_eq!(stats.votes_for, 1500);
    }
}
