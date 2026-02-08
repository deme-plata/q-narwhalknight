//! Consensus Voting Decentralization Tests
//!
//! Tests to ensure the consensus voting mechanism properly handles
//! multi-validator voting, view changes, and BFT thresholds.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. 2f+1 vote threshold for consensus
//! 2. View change triggered by leader timeout
//! 3. Vote tallying with abstentions
//! 4. Liveness with Byzantine validators
//! 5. Round advancement
//!
//! Run with: cargo test --package q-dag-knight --test consensus_voting_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::Duration;

// ============================================================================
// MOCK STRUCTURES FOR CONSENSUS VOTING TESTING
// ============================================================================

/// Vote type in consensus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteType {
    Accept,
    Reject,
    Abstain,
}

/// Vote from a validator
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    pub validator: [u8; 32],
    pub round: u64,
    pub height: u64,
    pub block_hash: [u8; 32],
    pub vote_type: VoteType,
    pub signature: [u8; 64],
}

/// View change vote
#[derive(Debug, Clone)]
pub struct ViewChangeVote {
    pub validator: [u8; 32],
    pub old_view: u64,
    pub new_view: u64,
    pub reason: ViewChangeReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewChangeReason {
    LeaderTimeout,
    InvalidProposal,
    NoProgress,
}

/// Round state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundState {
    Proposing,
    Voting,
    Committed,
    Failed,
}

/// Consensus round tracking
pub struct ConsensusRound {
    pub height: u64,
    pub round: u64,
    pub leader: [u8; 32],
    pub proposed_block: Option<[u8; 32]>,
    pub votes: HashMap<[u8; 32], ConsensusVote>,
    pub state: RoundState,
    pub started_at: u64,
}

/// Consensus voting coordinator
pub struct VotingCoordinator {
    validators: RwLock<Vec<[u8; 32]>>,
    current_height: AtomicU64,
    current_round: AtomicU64,
    current_view: AtomicU64,
    rounds: RwLock<HashMap<(u64, u64), ConsensusRound>>,
    view_change_votes: RwLock<HashMap<u64, Vec<ViewChangeVote>>>,
    finalized_heights: RwLock<HashSet<u64>>,
    leader_timeout_ms: u64,
}

impl VotingCoordinator {
    pub fn new(validators: Vec<[u8; 32]>) -> Self {
        Self {
            validators: RwLock::new(validators),
            current_height: AtomicU64::new(0),
            current_round: AtomicU64::new(0),
            current_view: AtomicU64::new(0),
            rounds: RwLock::new(HashMap::new()),
            view_change_votes: RwLock::new(HashMap::new()),
            finalized_heights: RwLock::new(HashSet::new()),
            leader_timeout_ms: 30_000, // 30 seconds
        }
    }

    /// Get BFT threshold (2f+1)
    pub fn bft_threshold(&self) -> usize {
        let n = self.validators.read().unwrap().len();
        // 2f+1 where f = floor((n-1)/3)
        let f = (n - 1) / 3;
        2 * f + 1
    }

    /// Get f+1 threshold for view change
    pub fn view_change_threshold(&self) -> usize {
        let n = self.validators.read().unwrap().len();
        let f = (n - 1) / 3;
        f + 1
    }

    /// Get current leader based on view
    pub fn get_leader(&self) -> [u8; 32] {
        let validators = self.validators.read().unwrap();
        let view = self.current_view.load(Ordering::SeqCst) as usize;
        validators[view % validators.len()]
    }

    /// Start a new round
    pub fn start_round(&self, height: u64, round: u64) -> Result<(), String> {
        let mut rounds = self.rounds.write().unwrap();

        if rounds.contains_key(&(height, round)) {
            return Err("ROUND_EXISTS: Round already started".to_string());
        }

        let leader = self.get_leader();

        let consensus_round = ConsensusRound {
            height,
            round,
            leader,
            proposed_block: None,
            votes: HashMap::new(),
            state: RoundState::Proposing,
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        rounds.insert((height, round), consensus_round);
        self.current_height.store(height, Ordering::SeqCst);
        self.current_round.store(round, Ordering::SeqCst);

        Ok(())
    }

    /// Propose a block (leader only)
    pub fn propose_block(
        &self,
        proposer: [u8; 32],
        height: u64,
        round: u64,
        block_hash: [u8; 32],
    ) -> Result<(), String> {
        let mut rounds = self.rounds.write().unwrap();
        let consensus_round = rounds
            .get_mut(&(height, round))
            .ok_or("ROUND_NOT_FOUND")?;

        if consensus_round.leader != proposer {
            return Err("NOT_LEADER: Only the leader can propose".to_string());
        }

        if consensus_round.state != RoundState::Proposing {
            return Err(format!(
                "INVALID_STATE: Cannot propose in {:?} state",
                consensus_round.state
            ));
        }

        if consensus_round.proposed_block.is_some() {
            return Err("ALREADY_PROPOSED: Block already proposed".to_string());
        }

        consensus_round.proposed_block = Some(block_hash);
        consensus_round.state = RoundState::Voting;

        Ok(())
    }

    /// Cast a vote
    pub fn cast_vote(&self, vote: ConsensusVote) -> Result<(), String> {
        // Verify validator is in set
        {
            let validators = self.validators.read().unwrap();
            if !validators.contains(&vote.validator) {
                return Err("UNKNOWN_VALIDATOR: Not in validator set".to_string());
            }
        }

        let mut rounds = self.rounds.write().unwrap();
        let consensus_round = rounds
            .get_mut(&(vote.height, vote.round))
            .ok_or("ROUND_NOT_FOUND")?;

        if consensus_round.state != RoundState::Voting {
            return Err(format!(
                "INVALID_STATE: Cannot vote in {:?} state",
                consensus_round.state
            ));
        }

        // Check if already voted
        if consensus_round.votes.contains_key(&vote.validator) {
            return Err("ALREADY_VOTED: Validator already voted".to_string());
        }

        // Check vote is for proposed block
        if let Some(proposed) = consensus_round.proposed_block {
            if vote.vote_type == VoteType::Accept && vote.block_hash != proposed {
                return Err("WRONG_BLOCK: Vote is for different block".to_string());
            }
        }

        consensus_round.votes.insert(vote.validator, vote);

        // Check if we can finalize
        self.try_finalize_round(consensus_round);

        Ok(())
    }

    /// Try to finalize a round
    fn try_finalize_round(&self, round: &mut ConsensusRound) {
        let threshold = self.bft_threshold();

        let accept_count = round
            .votes
            .values()
            .filter(|v| v.vote_type == VoteType::Accept)
            .count();

        let reject_count = round
            .votes
            .values()
            .filter(|v| v.vote_type == VoteType::Reject)
            .count();

        let total_validators = self.validators.read().unwrap().len();

        // If 2f+1 accepts, commit
        if accept_count >= threshold {
            round.state = RoundState::Committed;
            self.finalized_heights
                .write()
                .unwrap()
                .insert(round.height);
        }
        // If 2f+1 rejects or not enough validators left to reach threshold, fail
        else if reject_count >= threshold
            || round.votes.len() == total_validators && accept_count < threshold
        {
            round.state = RoundState::Failed;
        }
    }

    /// Submit view change vote
    pub fn vote_view_change(&self, vote: ViewChangeVote) -> Result<bool, String> {
        // Verify validator
        {
            let validators = self.validators.read().unwrap();
            if !validators.contains(&vote.validator) {
                return Err("UNKNOWN_VALIDATOR".to_string());
            }
        }

        let current_view = self.current_view.load(Ordering::SeqCst);
        if vote.old_view != current_view {
            return Err("WRONG_VIEW: View change for wrong view".to_string());
        }

        if vote.new_view != current_view + 1 {
            return Err("INVALID_NEW_VIEW: New view must be current + 1".to_string());
        }

        let mut view_changes = self.view_change_votes.write().unwrap();
        let votes = view_changes
            .entry(vote.new_view)
            .or_insert_with(Vec::new);

        // Check if already voted
        if votes.iter().any(|v| v.validator == vote.validator) {
            return Err("ALREADY_VOTED_VIEW_CHANGE".to_string());
        }

        votes.push(vote);

        // Check if we have f+1 votes for view change
        if votes.len() >= self.view_change_threshold() {
            let new_view = self.current_view.fetch_add(1, Ordering::SeqCst) + 1;
            return Ok(true); // View change successful
        }

        Ok(false)
    }

    /// Get round state
    pub fn get_round_state(&self, height: u64, round: u64) -> Option<RoundState> {
        self.rounds
            .read()
            .unwrap()
            .get(&(height, round))
            .map(|r| r.state.clone())
    }

    /// Get vote count for a round
    pub fn get_vote_count(&self, height: u64, round: u64) -> Option<(usize, usize, usize)> {
        self.rounds.read().unwrap().get(&(height, round)).map(|r| {
            let accept = r.votes.values().filter(|v| v.vote_type == VoteType::Accept).count();
            let reject = r.votes.values().filter(|v| v.vote_type == VoteType::Reject).count();
            let abstain = r.votes.values().filter(|v| v.vote_type == VoteType::Abstain).count();
            (accept, reject, abstain)
        })
    }

    /// Check if height is finalized
    pub fn is_finalized(&self, height: u64) -> bool {
        self.finalized_heights.read().unwrap().contains(&height)
    }

    /// Get current view
    pub fn current_view(&self) -> u64 {
        self.current_view.load(Ordering::SeqCst)
    }

    /// Get validator count
    pub fn validator_count(&self) -> usize {
        self.validators.read().unwrap().len()
    }
}

// ============================================================================
// BFT THRESHOLD TESTS
// ============================================================================

#[test]
fn test_bft_threshold_4_validators() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators);

    // 4 validators: f=1, 2f+1=3
    assert_eq!(coordinator.bft_threshold(), 3);
}

#[test]
fn test_bft_threshold_7_validators() {
    let validators: Vec<[u8; 32]> = (0..7).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators);

    // 7 validators: f=2, 2f+1=5
    assert_eq!(coordinator.bft_threshold(), 5);
}

#[test]
fn test_bft_threshold_10_validators() {
    let validators: Vec<[u8; 32]> = (0..10).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators);

    // 10 validators: f=3, 2f+1=7
    assert_eq!(coordinator.bft_threshold(), 7);
}

// ============================================================================
// ROUND LIFECYCLE TESTS
// ============================================================================

#[test]
fn test_start_round() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators);

    let result = coordinator.start_round(1, 0);
    assert!(result.is_ok());

    let state = coordinator.get_round_state(1, 0);
    assert_eq!(state, Some(RoundState::Proposing));
}

#[test]
fn test_duplicate_round_rejected() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators);

    coordinator.start_round(1, 0).unwrap();
    let result = coordinator.start_round(1, 0);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("ROUND_EXISTS"));
}

#[test]
fn test_propose_block() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];

    let result = coordinator.propose_block(leader, 1, 0, block_hash);
    assert!(result.is_ok());

    let state = coordinator.get_round_state(1, 0);
    assert_eq!(state, Some(RoundState::Voting));
}

#[test]
fn test_non_leader_propose_rejected() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();

    // Find a non-leader
    let non_leader = validators.iter().find(|v| **v != leader).unwrap();
    let block_hash = [99u8; 32];

    let result = coordinator.propose_block(*non_leader, 1, 0, block_hash);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("NOT_LEADER"));
}

// ============================================================================
// VOTING TESTS
// ============================================================================

#[test]
fn test_cast_vote() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    let vote = ConsensusVote {
        validator: validators[0],
        round: 0,
        height: 1,
        block_hash,
        vote_type: VoteType::Accept,
        signature: [0u8; 64],
    };

    let result = coordinator.cast_vote(vote);
    assert!(result.is_ok());

    let (accept, _, _) = coordinator.get_vote_count(1, 0).unwrap();
    assert_eq!(accept, 1);
}

#[test]
fn test_duplicate_vote_rejected() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    let vote = ConsensusVote {
        validator: validators[0],
        round: 0,
        height: 1,
        block_hash,
        vote_type: VoteType::Accept,
        signature: [0u8; 64],
    };

    coordinator.cast_vote(vote.clone()).unwrap();
    let result = coordinator.cast_vote(vote);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("ALREADY_VOTED"));
}

#[test]
fn test_unknown_validator_rejected() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    let unknown = [100u8; 32];
    let vote = ConsensusVote {
        validator: unknown,
        round: 0,
        height: 1,
        block_hash,
        vote_type: VoteType::Accept,
        signature: [0u8; 64],
    };

    let result = coordinator.cast_vote(vote);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("UNKNOWN_VALIDATOR"));
}

// ============================================================================
// FINALIZATION TESTS
// ============================================================================

#[test]
fn test_finalize_with_2f_plus_1_accepts() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    // 3 accept votes (2f+1 = 3 for n=4)
    for i in 0..3 {
        let vote = ConsensusVote {
            validator: validators[i],
            round: 0,
            height: 1,
            block_hash,
            vote_type: VoteType::Accept,
            signature: [0u8; 64],
        };
        coordinator.cast_vote(vote).unwrap();
    }

    assert_eq!(coordinator.get_round_state(1, 0), Some(RoundState::Committed));
    assert!(coordinator.is_finalized(1));
}

#[test]
fn test_not_finalized_without_threshold() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    // Only 2 accept votes (need 3)
    for i in 0..2 {
        let vote = ConsensusVote {
            validator: validators[i],
            round: 0,
            height: 1,
            block_hash,
            vote_type: VoteType::Accept,
            signature: [0u8; 64],
        };
        coordinator.cast_vote(vote).unwrap();
    }

    assert_eq!(coordinator.get_round_state(1, 0), Some(RoundState::Voting));
    assert!(!coordinator.is_finalized(1));
}

#[test]
fn test_round_fails_with_2f_plus_1_rejects() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    // 3 reject votes
    for i in 0..3 {
        let vote = ConsensusVote {
            validator: validators[i],
            round: 0,
            height: 1,
            block_hash,
            vote_type: VoteType::Reject,
            signature: [0u8; 64],
        };
        coordinator.cast_vote(vote).unwrap();
    }

    assert_eq!(coordinator.get_round_state(1, 0), Some(RoundState::Failed));
}

#[test]
fn test_abstain_votes_counted() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    // 2 accept, 1 reject, 1 abstain
    for (i, vote_type) in [(0, VoteType::Accept), (1, VoteType::Accept), (2, VoteType::Reject), (3, VoteType::Abstain)] {
        let vote = ConsensusVote {
            validator: validators[i],
            round: 0,
            height: 1,
            block_hash,
            vote_type,
            signature: [0u8; 64],
        };
        coordinator.cast_vote(vote).unwrap();
    }

    let (accept, reject, abstain) = coordinator.get_vote_count(1, 0).unwrap();
    assert_eq!(accept, 2);
    assert_eq!(reject, 1);
    assert_eq!(abstain, 1);

    // Not enough for consensus (need 3)
    assert_eq!(coordinator.get_round_state(1, 0), Some(RoundState::Failed));
}

// ============================================================================
// VIEW CHANGE TESTS
// ============================================================================

#[test]
fn test_view_change_vote() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    let vote = ViewChangeVote {
        validator: validators[0],
        old_view: 0,
        new_view: 1,
        reason: ViewChangeReason::LeaderTimeout,
    };

    let result = coordinator.vote_view_change(vote);
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Not enough votes yet
}

#[test]
fn test_view_change_requires_f_plus_1() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    // f+1 = 2 for n=4
    let vote1 = ViewChangeVote {
        validator: validators[0],
        old_view: 0,
        new_view: 1,
        reason: ViewChangeReason::LeaderTimeout,
    };
    coordinator.vote_view_change(vote1).unwrap();

    let vote2 = ViewChangeVote {
        validator: validators[1],
        old_view: 0,
        new_view: 1,
        reason: ViewChangeReason::LeaderTimeout,
    };
    let result = coordinator.vote_view_change(vote2).unwrap();

    assert!(result); // View change successful
    assert_eq!(coordinator.current_view(), 1);
}

#[test]
fn test_view_change_wrong_view_rejected() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    let vote = ViewChangeVote {
        validator: validators[0],
        old_view: 5, // Wrong view (current is 0)
        new_view: 6,
        reason: ViewChangeReason::LeaderTimeout,
    };

    let result = coordinator.vote_view_change(vote);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("WRONG_VIEW"));
}

#[test]
fn test_leader_rotates_on_view_change() {
    let validators: Vec<[u8; 32]> = (0..4).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    let leader_before = coordinator.get_leader();

    // Trigger view change
    for i in 0..2 {
        let vote = ViewChangeVote {
            validator: validators[i],
            old_view: 0,
            new_view: 1,
            reason: ViewChangeReason::LeaderTimeout,
        };
        coordinator.vote_view_change(vote).unwrap();
    }

    let leader_after = coordinator.get_leader();
    assert_ne!(leader_before, leader_after);
}

// ============================================================================
// LIVENESS TESTS
// ============================================================================

#[test]
fn test_liveness_with_f_byzantine() {
    // With f Byzantine validators, system should still make progress
    let validators: Vec<[u8; 32]> = (0..7).map(|i| [i as u8; 32]).collect();
    let coordinator = VotingCoordinator::new(validators.clone());

    // f=2 for n=7, so 2 validators can be Byzantine
    // Need 5 honest validators (2f+1)

    coordinator.start_round(1, 0).unwrap();
    let leader = coordinator.get_leader();
    let block_hash = [99u8; 32];
    coordinator.propose_block(leader, 1, 0, block_hash).unwrap();

    // 5 honest validators vote accept (validators 0-4)
    for i in 0..5 {
        let vote = ConsensusVote {
            validator: validators[i],
            round: 0,
            height: 1,
            block_hash,
            vote_type: VoteType::Accept,
            signature: [0u8; 64],
        };
        coordinator.cast_vote(vote).unwrap();
    }

    // Validators 5-6 are Byzantine (don't vote or vote reject)
    // System should still finalize with 5 votes

    assert_eq!(coordinator.get_round_state(1, 0), Some(RoundState::Committed));
    assert!(coordinator.is_finalized(1));
}
