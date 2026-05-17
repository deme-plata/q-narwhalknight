//! Committee Consensus
//!
//! Decentralized prediction verification with reputation and slashing.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Decentralized committee for prediction verification
pub struct PredictionCommittee {
    /// Committee members
    members: Vec<CommitteeMember>,

    /// Committee size target
    target_size: usize,

    /// Minimum stake for membership
    min_stake: u64,

    /// Reputation system
    reputation: ReputationSystem,

    /// BFT threshold (2/3 + 1)
    bft_threshold: f64,
}

/// Committee member
#[derive(Clone, Debug)]
pub struct CommitteeMember {
    /// Member address
    pub address: String,

    /// Staked amount
    pub stake: u64,

    /// Reputation score (0-1)
    pub reputation: f64,

    /// Last active block
    pub last_active: u64,

    /// Total verifications
    pub verifications: u64,

    /// Correct verifications
    pub correct_verifications: u64,
}

/// Reputation system for committee members
pub struct ReputationSystem {
    /// Reputation scores by address
    scores: HashMap<String, ReputationScore>,

    /// Decay rate per block
    decay_rate: f64,

    /// Last decay block
    last_decay_block: u64,
}

/// Individual reputation score
#[derive(Clone, Debug)]
struct ReputationScore {
    current: f64,
    history: Vec<ReputationEvent>,
    last_updated: u64,
}

#[derive(Clone, Debug)]
enum ReputationEvent {
    Increase(f64),
    Decrease(f64),
    Slash(f64),
}

impl Default for ReputationScore {
    fn default() -> Self {
        Self {
            current: 0.5, // Neutral starting reputation
            history: Vec::new(),
            last_updated: 0,
        }
    }
}

impl PredictionCommittee {
    /// Create new committee
    pub fn new(target_size: usize, min_stake: u64) -> Self {
        info!("🏛️ Creating prediction committee: size={}, min_stake={}",
              target_size, min_stake);

        Self {
            members: Vec::new(),
            target_size,
            min_stake,
            reputation: ReputationSystem::new(),
            bft_threshold: 0.67, // 2/3 majority
        }
    }

    /// Quick verification (simplified for Phase 1)
    pub async fn quick_verify(&self, _prediction: &crate::experts::ExpertPrediction) -> bool {
        // Phase 1: Always return true (verification comes in Phase 3)
        // In production, this would:
        // 1. Distribute prediction to committee members
        // 2. Collect signatures
        // 3. Verify BFT threshold reached

        if self.members.is_empty() {
            debug!("Committee empty - using default verification");
            return true;
        }

        // Simulate committee vote
        let approvals = self.simulate_committee_vote();
        let approval_ratio = approvals as f64 / self.members.len().max(1) as f64;

        approval_ratio >= self.bft_threshold
    }

    /// Simulate committee voting (placeholder for Phase 1)
    fn simulate_committee_vote(&self) -> usize {
        // In Phase 1, simulate based on member reputations
        self.members
            .iter()
            .filter(|m| m.reputation >= 0.5)
            .count()
    }

    /// Add member to committee
    pub fn add_member(&mut self, address: String, stake: u64) -> Result<(), CommitteeError> {
        // Validate stake
        if stake < self.min_stake {
            return Err(CommitteeError::InsufficientStake);
        }

        // Check if already member
        if self.members.iter().any(|m| m.address == address) {
            return Err(CommitteeError::AlreadyMember);
        }

        // Check committee size
        if self.members.len() >= self.target_size {
            // Would need to remove lowest reputation member
            if let Some(lowest_idx) = self.find_lowest_reputation_member() {
                let lowest_rep = self.members[lowest_idx].reputation;
                let new_rep = self.reputation.get_score(&address);

                if new_rep <= lowest_rep {
                    return Err(CommitteeError::CommitteeFull);
                }

                // Replace lowest reputation member
                self.members.remove(lowest_idx);
            }
        }

        let member = CommitteeMember {
            address: address.clone(),
            stake,
            reputation: self.reputation.get_score(&address),
            last_active: 0,
            verifications: 0,
            correct_verifications: 0,
        };

        self.members.push(member);
        info!("Added committee member: {}", address);

        Ok(())
    }

    /// Remove member from committee
    pub fn remove_member(&mut self, address: &str) {
        self.members.retain(|m| m.address != address);
    }

    /// Find lowest reputation member
    fn find_lowest_reputation_member(&self) -> Option<usize> {
        self.members
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.reputation.partial_cmp(&b.reputation).unwrap()
            })
            .map(|(idx, _)| idx)
    }

    /// Update member after verification
    pub fn record_verification(
        &mut self,
        address: &str,
        was_correct: bool,
        current_block: u64,
    ) {
        if let Some(member) = self.members.iter_mut().find(|m| m.address == address) {
            member.verifications += 1;
            member.last_active = current_block;

            if was_correct {
                member.correct_verifications += 1;
                self.reputation.increase(address, 0.01);
            } else {
                self.reputation.decrease(address, 0.05);
            }

            member.reputation = self.reputation.get_score(address);
        }
    }

    /// Slash member for malicious behavior
    pub fn slash_member(&mut self, address: &str, percentage: f64) {
        if let Some(member) = self.members.iter_mut().find(|m| m.address == address) {
            let slash_amount = (member.stake as f64 * percentage) as u64;
            member.stake = member.stake.saturating_sub(slash_amount);

            self.reputation.slash(address, 0.2);
            member.reputation = self.reputation.get_score(address);

            warn!("Slashed member {} by {}% ({} stake)", address, percentage * 100.0, slash_amount);

            // Remove if stake falls below minimum
            if member.stake < self.min_stake {
                self.remove_member(address);
                warn!("Removed member {} due to insufficient stake after slash", address);
            }
        }
    }

    /// Get committee members
    pub fn members(&self) -> &[CommitteeMember] {
        &self.members
    }

    /// Get committee size
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

impl ReputationSystem {
    /// Create new reputation system
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            decay_rate: 0.001, // Decay towards 0.5 over time
            last_decay_block: 0,
        }
    }

    /// Get reputation score
    pub fn get_score(&self, address: &str) -> f64 {
        self.scores
            .get(address)
            .map(|s| s.current)
            .unwrap_or(0.5)
    }

    /// Increase reputation
    pub fn increase(&mut self, address: &str, amount: f64) {
        let score = self.scores.entry(address.to_string())
            .or_insert(ReputationScore::default());

        score.current = (score.current + amount).min(1.0);
        score.history.push(ReputationEvent::Increase(amount));
    }

    /// Decrease reputation
    pub fn decrease(&mut self, address: &str, amount: f64) {
        let score = self.scores.entry(address.to_string())
            .or_insert(ReputationScore::default());

        score.current = (score.current - amount).max(0.0);
        score.history.push(ReputationEvent::Decrease(amount));
    }

    /// Slash reputation (severe decrease)
    pub fn slash(&mut self, address: &str, amount: f64) {
        let score = self.scores.entry(address.to_string())
            .or_insert(ReputationScore::default());

        score.current = (score.current - amount).max(0.0);
        score.history.push(ReputationEvent::Slash(amount));
    }

    /// Apply time-based decay
    pub fn apply_decay(&mut self, current_block: u64) {
        let blocks_passed = current_block.saturating_sub(self.last_decay_block);

        if blocks_passed == 0 {
            return;
        }

        let decay_factor = (1.0 - self.decay_rate).powi(blocks_passed as i32);

        for score in self.scores.values_mut() {
            // Decay towards neutral (0.5)
            score.current = 0.5 + (score.current - 0.5) * decay_factor;
        }

        self.last_decay_block = current_block;
    }
}

impl Default for ReputationSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Committee errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum CommitteeError {
    #[error("Insufficient stake for committee membership")]
    InsufficientStake,

    #[error("Already a committee member")]
    AlreadyMember,

    #[error("Committee is full")]
    CommitteeFull,

    #[error("Not a committee member")]
    NotMember,

    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_committee_creation() {
        let committee = PredictionCommittee::new(21, 10_000_000_000);
        assert_eq!(committee.size(), 0);
        assert_eq!(committee.target_size, 21);
    }

    #[test]
    fn test_add_member() {
        let mut committee = PredictionCommittee::new(21, 10_000_000_000);

        committee.add_member("addr1".to_string(), 10_000_000_000).unwrap();
        assert_eq!(committee.size(), 1);

        // Duplicate should fail
        let result = committee.add_member("addr1".to_string(), 10_000_000_000);
        assert!(result.is_err());

        // Insufficient stake should fail
        let result = committee.add_member("addr2".to_string(), 1_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_reputation_system() {
        let mut rep = ReputationSystem::new();

        assert!((rep.get_score("addr1") - 0.5).abs() < 1e-10);

        rep.increase("addr1", 0.1);
        assert!(rep.get_score("addr1") > 0.5);

        rep.decrease("addr1", 0.05);
        assert!(rep.get_score("addr1") < 0.6);

        rep.slash("addr1", 0.5);
        assert!(rep.get_score("addr1") < 0.1);
    }

    #[tokio::test]
    async fn test_quick_verify() {
        let committee = PredictionCommittee::new(21, 10_000_000_000);

        // Empty committee should return true (Phase 1 default)
        let prediction = crate::experts::ExpertPrediction {
            primary_value: 0.5,
            confidence: 0.8,
            domain: crate::experts::PredictionDomain::FeeForecasting,
            expert_weights: vec![],
            quantum_entropy: 0.1,
        };

        assert!(committee.quick_verify(&prediction).await);
    }
}
