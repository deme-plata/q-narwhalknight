//! Governance module for QNK-INDEX
//!
//! Enables decentralized management of index funds through
//! shareholder voting on proposals.

use crate::types::*;
use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Minimum voting period in blocks (~24 hours at 6s blocks)
pub const MIN_VOTING_PERIOD: u64 = 14400;

/// Maximum voting period in blocks (~7 days)
pub const MAX_VOTING_PERIOD: u64 = 100800;

/// Quorum required for proposal to pass (basis points)
pub const DEFAULT_QUORUM_BPS: u16 = 1000; // 10% of total shares

/// Approval threshold (basis points)
pub const DEFAULT_APPROVAL_BPS: u16 = 5000; // 50% of votes

/// Governance manager
pub struct GovernanceManager {
    /// All proposals: (index_id, proposal_id) -> Proposal
    proposals: DashMap<([u8; 32], u64), GovernanceProposal>,

    /// Vote records: (index_id, proposal_id, voter) -> (shares, for_or_against)
    votes: DashMap<([u8; 32], u64, [u8; 32]), (u64, bool)>,

    /// Proposal counter per index
    proposal_counters: DashMap<[u8; 32], u64>,

    /// Current block height
    current_block: Arc<RwLock<u64>>,

    /// Minimum shares to create proposal (basis points of total supply)
    min_proposal_shares_bps: u16,
}

impl GovernanceManager {
    /// Create new governance manager
    pub fn new() -> Self {
        Self {
            proposals: DashMap::new(),
            votes: DashMap::new(),
            proposal_counters: DashMap::new(),
            current_block: Arc::new(RwLock::new(0)),
            min_proposal_shares_bps: 100, // 1% of supply to propose
        }
    }

    /// Set current block height
    pub async fn set_block_height(&self, height: u64) {
        let mut block = self.current_block.write().await;
        *block = height;
    }

    /// Get current block height
    pub async fn get_block_height(&self) -> u64 {
        *self.current_block.read().await
    }

    /// Create a new proposal
    pub async fn create_proposal(
        &self,
        index: &QnkIndex,
        proposer: [u8; 32],
        proposer_shares: u64,
        proposal_type: ProposalType,
        description: String,
        voting_period_blocks: u64,
    ) -> Result<u64, IndexError> {
        // Check governance is enabled
        if !index.governance_enabled {
            return Err(IndexError::GovernanceError("Governance not enabled".into()));
        }

        // Check proposer has enough shares
        let min_shares = index.total_supply * self.min_proposal_shares_bps as u64 / 10000;
        if proposer_shares < min_shares {
            return Err(IndexError::GovernanceError(
                format!("Need {} shares to propose, have {}", min_shares, proposer_shares)
            ));
        }

        // Validate voting period
        if voting_period_blocks < MIN_VOTING_PERIOD || voting_period_blocks > MAX_VOTING_PERIOD {
            return Err(IndexError::GovernanceError("Invalid voting period".into()));
        }

        // Get next proposal ID
        let proposal_id = {
            let mut counter = self.proposal_counters.entry(index.index_id).or_insert(0);
            *counter += 1;
            *counter
        };

        let current_block = self.get_block_height().await;

        let proposal = GovernanceProposal {
            proposal_id,
            index_id: index.index_id,
            proposer,
            proposal_type: proposal_type.clone(),
            description,
            start_block: current_block,
            end_block: current_block + voting_period_blocks,
            votes_for: 0,
            votes_against: 0,
            voters: Vec::new(),
            status: ProposalStatus::Active,
            execution_data: self.encode_proposal_data(&proposal_type),
        };

        self.proposals.insert((index.index_id, proposal_id), proposal);

        info!(
            "Created proposal {} for index {}: {:?}",
            proposal_id,
            hex::encode(&index.index_id[..8]),
            proposal_type
        );

        Ok(proposal_id)
    }

    /// Cast a vote on a proposal
    pub async fn vote(
        &self,
        index_id: [u8; 32],
        proposal_id: u64,
        voter: [u8; 32],
        voter_shares: u64,
        support: bool,
    ) -> Result<(), IndexError> {
        let current_block = self.get_block_height().await;

        // Get and update proposal
        let mut proposal = self.proposals.get_mut(&(index_id, proposal_id))
            .ok_or(IndexError::ProposalNotFound)?;

        // Check voting is still active
        if proposal.status != ProposalStatus::Active {
            return Err(IndexError::GovernanceError("Proposal not active".into()));
        }

        if current_block > proposal.end_block {
            return Err(IndexError::VotingEnded);
        }

        // Check not already voted
        if proposal.voters.contains(&voter) {
            return Err(IndexError::AlreadyVoted);
        }

        // Record vote
        if support {
            proposal.votes_for = proposal.votes_for
                .checked_add(voter_shares)
                .ok_or(IndexError::ArithmeticOverflow)?;
        } else {
            proposal.votes_against = proposal.votes_against
                .checked_add(voter_shares)
                .ok_or(IndexError::ArithmeticOverflow)?;
        }

        proposal.voters.push(voter);

        // Store vote record
        self.votes.insert((index_id, proposal_id, voter), (voter_shares, support));

        debug!(
            "Vote cast on proposal {}: {} shares {}",
            proposal_id,
            voter_shares,
            if support { "FOR" } else { "AGAINST" }
        );

        Ok(())
    }

    /// Finalize a proposal (after voting ends)
    pub async fn finalize_proposal(
        &self,
        index_id: [u8; 32],
        proposal_id: u64,
        total_supply: u64,
    ) -> Result<ProposalStatus, IndexError> {
        let current_block = self.get_block_height().await;

        let mut proposal = self.proposals.get_mut(&(index_id, proposal_id))
            .ok_or(IndexError::ProposalNotFound)?;

        // Check voting has ended
        if current_block < proposal.end_block {
            return Err(IndexError::GovernanceError("Voting not yet ended".into()));
        }

        // Already finalized?
        if proposal.status != ProposalStatus::Active {
            return Ok(proposal.status.clone());
        }

        // Check quorum
        let total_votes = proposal.votes_for + proposal.votes_against;
        let quorum_needed = total_supply * DEFAULT_QUORUM_BPS as u64 / 10000;

        if total_votes < quorum_needed {
            proposal.status = ProposalStatus::Rejected;
            warn!(
                "Proposal {} rejected: quorum not met ({}/{})",
                proposal_id, total_votes, quorum_needed
            );
            return Ok(ProposalStatus::Rejected);
        }

        // Check approval threshold
        let approval_needed = total_votes * DEFAULT_APPROVAL_BPS as u64 / 10000;

        if proposal.votes_for >= approval_needed {
            proposal.status = ProposalStatus::Passed;
            info!(
                "Proposal {} passed: {} for, {} against",
                proposal_id, proposal.votes_for, proposal.votes_against
            );
            Ok(ProposalStatus::Passed)
        } else {
            proposal.status = ProposalStatus::Rejected;
            info!(
                "Proposal {} rejected: {} for, {} against",
                proposal_id, proposal.votes_for, proposal.votes_against
            );
            Ok(ProposalStatus::Rejected)
        }
    }

    /// Execute a passed proposal
    pub async fn execute_proposal(
        &self,
        index: &mut QnkIndex,
        proposal_id: u64,
    ) -> Result<(), IndexError> {
        let mut proposal = self.proposals.get_mut(&(index.index_id, proposal_id))
            .ok_or(IndexError::ProposalNotFound)?;

        // Must be passed
        if proposal.status != ProposalStatus::Passed {
            return Err(IndexError::GovernanceError("Proposal not passed".into()));
        }

        // Execute based on type
        match &proposal.proposal_type {
            ProposalType::AddComponent { token_address, symbol, initial_weight_bps } => {
                if index.components.len() >= index.max_components as usize {
                    return Err(IndexError::IndexFull);
                }

                let component = IndexComponent {
                    token_address: *token_address,
                    symbol: symbol.clone(),
                    target_weight_bps: *initial_weight_bps,
                    actual_weight_bps: 0,
                    holdings: 0,
                    price_qug: 100_000_000,
                    rank: (index.components.len() + 1) as u8,
                };

                index.components.push(component);
            }

            ProposalType::RemoveComponent { token_address } => {
                index.components.retain(|c| c.token_address != *token_address);
                // Rerank
                for (i, comp) in index.components.iter_mut().enumerate() {
                    comp.rank = (i + 1) as u8;
                }
            }

            ProposalType::ChangeWeights { new_weights } => {
                for (rank, new_weight) in new_weights {
                    if let Some(comp) = index.components.iter_mut()
                        .find(|c| c.rank == *rank)
                    {
                        comp.target_weight_bps = *new_weight;
                    }
                }
            }

            ProposalType::ChangeFee { new_fee_bps } => {
                if *new_fee_bps > 500 {
                    return Err(IndexError::FeeTooHigh);
                }
                index.management_fee_bps = *new_fee_bps;
            }

            ProposalType::ChangeManager { new_manager } => {
                index.manager = *new_manager;
            }

            ProposalType::EmergencyPause { pause_mint, pause_redeem, pause_rebalance } => {
                index.paused_mint = *pause_mint;
                index.paused_redeem = *pause_redeem;
                index.paused_rebalance = *pause_rebalance;

                if *pause_mint || *pause_redeem || *pause_rebalance {
                    index.emergency_paused_at = Some(self.get_block_height().await);
                } else {
                    index.emergency_paused_at = None;
                }
            }

            ProposalType::ChangeMethodology { new_methodology } => {
                // Validate new methodology
                crate::QnkIndexFund::validate_methodology(new_methodology)?;
                index.methodology = new_methodology.clone();
            }
        }

        proposal.status = ProposalStatus::Executed;

        info!(
            "Executed proposal {} on index {}",
            proposal_id,
            hex::encode(&index.index_id[..8])
        );

        Ok(())
    }

    /// Cancel a proposal (only by proposer before voting ends)
    pub async fn cancel_proposal(
        &self,
        index_id: [u8; 32],
        proposal_id: u64,
        caller: [u8; 32],
    ) -> Result<(), IndexError> {
        let mut proposal = self.proposals.get_mut(&(index_id, proposal_id))
            .ok_or(IndexError::ProposalNotFound)?;

        // Only proposer can cancel
        if proposal.proposer != caller {
            return Err(IndexError::Unauthorized);
        }

        // Can only cancel if still active
        if proposal.status != ProposalStatus::Active {
            return Err(IndexError::GovernanceError("Cannot cancel non-active proposal".into()));
        }

        proposal.status = ProposalStatus::Cancelled;

        info!("Proposal {} cancelled by proposer", proposal_id);

        Ok(())
    }

    /// Get proposal info
    pub fn get_proposal(&self, index_id: [u8; 32], proposal_id: u64) -> Option<GovernanceProposal> {
        self.proposals.get(&(index_id, proposal_id)).map(|p| p.clone())
    }

    /// Get all active proposals for an index
    pub fn get_active_proposals(&self, index_id: [u8; 32]) -> Vec<GovernanceProposal> {
        self.proposals.iter()
            .filter(|p| p.key().0 == index_id && p.value().status == ProposalStatus::Active)
            .map(|p| p.value().clone())
            .collect()
    }

    /// Get voter's vote on a proposal
    pub fn get_vote(
        &self,
        index_id: [u8; 32],
        proposal_id: u64,
        voter: [u8; 32],
    ) -> Option<(u64, bool)> {
        self.votes.get(&(index_id, proposal_id, voter)).map(|v| *v)
    }

    /// Encode proposal data for on-chain storage
    fn encode_proposal_data(&self, proposal_type: &ProposalType) -> Vec<u8> {
        bincode::serialize(proposal_type).unwrap_or_default()
    }
}

impl Default for GovernanceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proposal_lifecycle() {
        let gov = GovernanceManager::new();
        gov.set_block_height(1000).await;

        let index = QnkIndex {
            index_id: [1u8; 32],
            name: "Test".into(),
            symbol: "TST".into(),
            total_supply: 1_000_000,
            components: vec![],
            nav_per_share: 100_000_000,
            last_rebalance_height: 0,
            rebalance_interval: 1000,
            management_fee_bps: 100,
            performance_fee_bps: 0,
            min_market_cap: 0,
            max_components: 10,
            manager: [0u8; 32],
            governance_enabled: true,
            methodology: IndexMethodology::EqualWeight { components_count: 10 },
            creation_block: 0,
            total_fees_accrued: 0,
            high_water_mark: 100_000_000,
            paused_mint: false,
            paused_redeem: false,
            paused_rebalance: false,
            emergency_paused_at: None,
        };

        let proposer = [2u8; 32];
        let proposer_shares = 50_000; // 5% of supply

        // Create proposal
        let proposal_id = gov.create_proposal(
            &index,
            proposer,
            proposer_shares,
            ProposalType::ChangeFee { new_fee_bps: 50 },
            "Lower fees".into(),
            MIN_VOTING_PERIOD,
        ).await.unwrap();

        assert_eq!(proposal_id, 1);

        // Vote
        let voter1 = [3u8; 32];
        gov.vote(index.index_id, proposal_id, voter1, 200_000, true).await.unwrap();

        let voter2 = [4u8; 32];
        gov.vote(index.index_id, proposal_id, voter2, 100_000, false).await.unwrap();

        // Finalize after voting ends
        gov.set_block_height(1000 + MIN_VOTING_PERIOD + 1).await;

        let result = gov.finalize_proposal(index.index_id, proposal_id, index.total_supply).await.unwrap();
        assert_eq!(result, ProposalStatus::Passed);
    }

    #[tokio::test]
    async fn test_quorum_not_met() {
        let gov = GovernanceManager::new();
        gov.set_block_height(1000).await;

        let index = QnkIndex {
            index_id: [5u8; 32],
            name: "Test2".into(),
            symbol: "TS2".into(),
            total_supply: 1_000_000,
            components: vec![],
            nav_per_share: 100_000_000,
            last_rebalance_height: 0,
            rebalance_interval: 1000,
            management_fee_bps: 100,
            performance_fee_bps: 0,
            min_market_cap: 0,
            max_components: 10,
            manager: [0u8; 32],
            governance_enabled: true,
            methodology: IndexMethodology::EqualWeight { components_count: 10 },
            creation_block: 0,
            total_fees_accrued: 0,
            high_water_mark: 100_000_000,
            paused_mint: false,
            paused_redeem: false,
            paused_rebalance: false,
            emergency_paused_at: None,
        };

        let proposer = [6u8; 32];
        let proposal_id = gov.create_proposal(
            &index,
            proposer,
            20_000,
            ProposalType::ChangeFee { new_fee_bps: 50 },
            "Lower fees".into(),
            MIN_VOTING_PERIOD,
        ).await.unwrap();

        // Only small vote
        let voter = [7u8; 32];
        gov.vote(index.index_id, proposal_id, voter, 5_000, true).await.unwrap();

        // Finalize - should fail quorum
        gov.set_block_height(1000 + MIN_VOTING_PERIOD + 1).await;

        let result = gov.finalize_proposal(index.index_id, proposal_id, index.total_supply).await.unwrap();
        assert_eq!(result, ProposalStatus::Rejected);
    }
}
