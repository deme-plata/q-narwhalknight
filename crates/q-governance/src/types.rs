//! Core types for Proof-of-Contribution Governance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Governance proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Unique proposal ID
    pub id: String,

    /// Proposal title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Address of proposer
    pub proposer: [u8; 32],

    /// Voting options
    pub options: Vec<String>,

    /// Voting start time (Unix timestamp)
    pub voting_start: u64,

    /// Voting end time (Unix timestamp)
    pub voting_end: u64,

    /// Type of proposal
    pub proposal_type: ProposalType,

    /// Required quorum (minimum voting power)
    pub required_quorum: u128,
}

/// Types of governance proposals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalType {
    /// Change protocol parameters (fees, limits, etc.)
    ParameterChange,

    /// Allocate treasury funds
    TreasuryAllocation,

    /// Protocol upgrade
    ProtocolUpgrade,

    /// General governance decision
    General,
}

/// Weighted governance vote with optional mining contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedVote {
    /// Proposal being voted on
    pub proposal_id: String,

    /// Voter's address
    pub voter_address: [u8; 32],

    /// Vote choice (must match proposal option)
    pub vote_choice: String,

    /// Token stake voting power
    pub token_stake: u128,

    /// Optional mining contribution (increases voting power)
    pub mining_contribution: Option<MiningContribution>,

    /// Timestamp of vote
    pub timestamp: u64,

    /// AEGIS-QL signature of vote
    pub signature: Vec<u8>,
}

/// Mining contribution to increase voting power
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningContribution {
    /// Mining solutions contributed (can be empty if using historical data)
    pub solutions: Vec<MiningSolutionRef>,

    /// Total hashes performed
    pub total_hashes: u128,

    /// Blockchain proofs of mining work
    pub merkle_proofs: Vec<MerkleProof>,

    /// Time period of contribution (start, end timestamps)
    pub contribution_period: (u64, u64),
}

/// Reference to a mining solution in the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningSolutionRef {
    /// Block height containing this solution
    pub block_height: u64,

    /// Index of solution in block
    pub solution_index: usize,

    /// Hash of the solution for quick lookup
    pub solution_hash: [u8; 32],
}

/// Merkle proof linking mining work to blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Merkle root in block header
    pub root: [u8; 32],

    /// Block height
    pub block_height: u64,

    /// Merkle path
    pub path: Vec<[u8; 32]>,

    /// Position in merkle tree
    pub position: usize,
}

/// Result of a governance vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalResult {
    /// Proposal ID
    pub proposal_id: String,

    /// Total number of votes cast
    pub total_votes: usize,

    /// Total voting power used
    pub total_voting_power: u128,

    /// Voting power per option
    pub option_results: HashMap<String, u128>,

    /// Winning option (if any)
    pub winning_option: Option<(String, u128)>,

    /// Whether result is finalized
    pub finalized: bool,
}

/// Statistics about mining contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionStats {
    /// Total hashes contributed
    pub total_hashes: u128,

    /// Number of mining solutions
    pub solution_count: u64,

    /// Time period covered
    pub period: (u64, u64),

    /// Estimated voting power bonus (percentage)
    pub power_bonus_percent: f64,
}
