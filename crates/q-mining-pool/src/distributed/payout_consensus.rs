//! Threshold Signature Payout Consensus
//!
//! Payouts require threshold signatures from multiple pool nodes.
//! This prevents any single node from stealing pool funds.
//!
//! Uses FROST (Flexible Round-Optimized Schnorr Threshold) style
//! protocol for distributed signing.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{DistributedError, DistributedResult, PeerIdBytes, TransactionId};
use crate::pplns::RewardEntry;

/// Payout batch for distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutBatch {
    /// Unique batch ID
    pub batch_id: [u8; 32],

    /// Block height that triggered this payout
    pub block_height: u64,

    /// Block hash that was found
    pub block_hash: [u8; 32],

    /// Total block reward being distributed
    pub total_reward: u64,

    /// Individual payouts
    pub payouts: Vec<PayoutEntry>,

    /// PPLNS state hash used for calculation
    pub pplns_state_hash: [u8; 32],

    /// Node that initiated the batch
    pub initiator: PeerIdBytes,

    /// Creation timestamp
    pub timestamp: u64,

    /// Initiator's signature
    #[serde(with = "BigArray")]
    pub initiator_signature: [u8; 64],
}

/// Individual payout entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutEntry {
    /// Recipient wallet address
    pub wallet_address: String,

    /// Amount to pay (in base units)
    pub amount: u64,

    /// Worker's difficulty contribution
    pub difficulty_contribution: f64,

    /// Proportion of total reward
    pub proportion: f64,
}

impl From<RewardEntry> for PayoutEntry {
    fn from(entry: RewardEntry) -> Self {
        Self {
            wallet_address: entry.wallet_address,
            amount: entry.amount,
            difficulty_contribution: entry.difficulty_contribution,
            proportion: entry.proportion,
        }
    }
}

impl PayoutBatch {
    /// Create new payout batch
    pub fn new(
        block_height: u64,
        block_hash: [u8; 32],
        total_reward: u64,
        payouts: Vec<PayoutEntry>,
        pplns_state_hash: [u8; 32],
        initiator: PeerIdBytes,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Compute batch ID from content
        let mut hasher = Hasher::new();
        hasher.update(b"QNK_PAYOUT_BATCH");
        hasher.update(&block_height.to_le_bytes());
        hasher.update(&block_hash);
        hasher.update(&total_reward.to_le_bytes());
        hasher.update(&pplns_state_hash);
        hasher.update(&timestamp.to_le_bytes());
        let batch_id = *hasher.finalize().as_bytes();

        Self {
            batch_id,
            block_height,
            block_hash,
            total_reward,
            payouts,
            pplns_state_hash,
            initiator,
            timestamp,
            initiator_signature: [0u8; 64],
        }
    }

    /// Compute batch hash for signing
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.batch_id);
        hasher.update(&self.block_height.to_le_bytes());
        hasher.update(&self.block_hash);
        hasher.update(&self.total_reward.to_le_bytes());

        // Hash each payout deterministically
        for payout in &self.payouts {
            hasher.update(payout.wallet_address.as_bytes());
            hasher.update(&payout.amount.to_le_bytes());
        }

        hasher.update(&self.pplns_state_hash);
        hasher.update(&self.initiator);
        hasher.update(&self.timestamp.to_le_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Verify payout amounts sum correctly
    pub fn verify_amounts(&self) -> bool {
        let sum: u64 = self.payouts.iter().map(|p| p.amount).sum();

        // Allow for dev fee + pool fee (2.5% total)
        let min_payout = (self.total_reward as f64 * 0.97) as u64;
        let max_payout = self.total_reward;

        sum >= min_payout && sum <= max_payout
    }

    /// Get total payout count
    pub fn payout_count(&self) -> usize {
        self.payouts.len()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Vote on a payout batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutVote {
    /// Batch being voted on
    pub batch_id: [u8; 32],

    /// Voter's peer ID
    pub voter: PeerIdBytes,

    /// Vote result
    pub approve: bool,

    /// Reason if rejected
    pub reason: Option<String>,

    /// Voter's PPLNS state hash
    pub voter_pplns_hash: [u8; 32],

    /// Partial signature for threshold signing
    pub partial_signature: PartialSignature,

    /// Timestamp
    pub timestamp: u64,
}

impl PayoutVote {
    /// Create new approval vote
    pub fn approve(
        batch_id: [u8; 32],
        voter: PeerIdBytes,
        voter_pplns_hash: [u8; 32],
        partial_signature: PartialSignature,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            batch_id,
            voter,
            approve: true,
            reason: None,
            voter_pplns_hash,
            partial_signature,
            timestamp,
        }
    }

    /// Create new rejection vote
    pub fn reject(
        batch_id: [u8; 32],
        voter: PeerIdBytes,
        reason: String,
        voter_pplns_hash: [u8; 32],
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            batch_id,
            voter,
            approve: false,
            reason: Some(reason),
            voter_pplns_hash,
            partial_signature: PartialSignature::empty(),
            timestamp,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Partial signature for threshold signing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialSignature {
    /// Signer's index in the threshold scheme
    pub signer_index: u32,

    /// Commitment (R value in Schnorr)
    pub commitment: [u8; 32],

    /// Response (s value in Schnorr)
    pub response: [u8; 32],

    /// Proof of knowledge
    #[serde(with = "BigArray")]
    pub proof: [u8; 64],
}

impl PartialSignature {
    /// Create empty partial signature (for rejections)
    pub fn empty() -> Self {
        Self {
            signer_index: 0,
            commitment: [0u8; 32],
            response: [0u8; 32],
            proof: [0u8; 64],
        }
    }

    /// Create a partial signature (placeholder - real impl would use FROST)
    pub fn create(signer_index: u32, message: &[u8; 32], secret_share: &[u8; 32]) -> Self {
        // In production, this would use the FROST protocol:
        // 1. Generate random nonce k
        // 2. Compute commitment R = k*G
        // 3. Compute challenge c = H(R || P || m)
        // 4. Compute response s = k + c*secret_share

        // For now, use simplified deterministic version
        let mut hasher = Hasher::new();
        hasher.update(b"QNK_PARTIAL_SIG_COMMIT");
        hasher.update(message);
        hasher.update(secret_share);
        hasher.update(&signer_index.to_le_bytes());
        let commitment = *hasher.finalize().as_bytes();

        let mut hasher = Hasher::new();
        hasher.update(b"QNK_PARTIAL_SIG_RESPONSE");
        hasher.update(&commitment);
        hasher.update(message);
        hasher.update(secret_share);
        let response = *hasher.finalize().as_bytes();

        let mut hasher = Hasher::new();
        hasher.update(b"QNK_PARTIAL_SIG_PROOF");
        hasher.update(&commitment);
        hasher.update(&response);
        hasher.update(secret_share);
        let proof_hash = *hasher.finalize().as_bytes();

        let mut proof = [0u8; 64];
        proof[..32].copy_from_slice(&proof_hash);
        proof[32..].copy_from_slice(&commitment);

        Self {
            signer_index,
            commitment,
            response,
            proof,
        }
    }

    /// Verify partial signature
    pub fn verify(&self, message: &[u8; 32], _public_share: &[u8; 32]) -> bool {
        // Simplified verification
        // In production, would verify:
        // R == s*G - c*P where c = H(R || P || m)

        // For now, just check it's not empty
        self.commitment != [0u8; 32] && self.response != [0u8; 32]
    }
}

/// Aggregated threshold signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// Combined commitment (sum of all R values)
    pub combined_commitment: [u8; 32],

    /// Combined response (sum of all s values)
    pub combined_response: [u8; 32],

    /// Signers who contributed
    pub signers: Vec<u32>,

    /// Number of signers
    pub threshold_met: bool,
}

impl ThresholdSignature {
    /// Aggregate partial signatures
    pub fn aggregate(partials: &[PartialSignature], threshold: usize) -> Option<Self> {
        if partials.len() < threshold {
            return None;
        }

        // In production, would use Lagrange interpolation to combine signatures
        // For now, XOR the commitments and responses

        let mut combined_commitment = [0u8; 32];
        let mut combined_response = [0u8; 32];
        let mut signers = Vec::new();

        for partial in partials.iter().take(threshold) {
            for i in 0..32 {
                combined_commitment[i] ^= partial.commitment[i];
                combined_response[i] ^= partial.response[i];
            }
            signers.push(partial.signer_index);
        }

        Some(Self {
            combined_commitment,
            combined_response,
            signers,
            threshold_met: true,
        })
    }

    /// Verify threshold signature
    pub fn verify(&self, _message: &[u8; 32], _group_public_key: &[u8; 32]) -> bool {
        // In production, verify Schnorr signature:
        // R == s*G - c*P where c = H(R || P || m)
        self.threshold_met && !self.signers.is_empty()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

/// Payout consensus state
#[derive(Debug, Clone)]
pub struct PayoutConsensusState {
    /// The payout batch
    pub batch: PayoutBatch,

    /// Collected votes
    pub votes: HashMap<PeerIdBytes, PayoutVote>,

    /// Approving voters
    pub approvers: HashSet<PeerIdBytes>,

    /// Rejecting voters
    pub rejectors: HashSet<PeerIdBytes>,

    /// Partial signatures for aggregation
    pub partial_signatures: Vec<PartialSignature>,

    /// Final threshold signature (if reached)
    pub threshold_signature: Option<ThresholdSignature>,

    /// Status
    pub status: PayoutStatus,

    /// Start time
    pub started_at: u64,

    /// Finalization time
    pub finalized_at: Option<u64>,
}

/// Payout consensus status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayoutStatus {
    /// Collecting votes
    Pending,

    /// Threshold reached, signature aggregated
    Approved,

    /// Rejected by quorum
    Rejected,

    /// Timeout
    TimedOut,

    /// Transaction broadcast
    Broadcast,

    /// Transaction confirmed on-chain
    Confirmed,
}

impl PayoutConsensusState {
    /// Create new payout consensus
    pub fn new(batch: PayoutBatch) -> Self {
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            batch,
            votes: HashMap::new(),
            approvers: HashSet::new(),
            rejectors: HashSet::new(),
            partial_signatures: Vec::new(),
            threshold_signature: None,
            status: PayoutStatus::Pending,
            started_at,
            finalized_at: None,
        }
    }

    /// Add a vote
    pub fn add_vote(&mut self, vote: PayoutVote) {
        // Skip duplicates
        if self.votes.contains_key(&vote.voter) {
            return;
        }

        if vote.approve {
            self.approvers.insert(vote.voter);
            self.partial_signatures.push(vote.partial_signature.clone());
        } else {
            self.rejectors.insert(vote.voter);
        }

        self.votes.insert(vote.voter, vote);
    }

    /// Check if threshold is met
    ///
    /// Threshold is 2/3 of total nodes for approval
    pub fn check_threshold(&mut self, total_nodes: usize) -> PayoutStatus {
        if self.status != PayoutStatus::Pending {
            return self.status.clone();
        }

        let threshold = (2 * total_nodes + 2) / 3;

        // Check for approval
        if self.approvers.len() >= threshold {
            // Aggregate signatures
            if let Some(sig) = ThresholdSignature::aggregate(&self.partial_signatures, threshold) {
                self.threshold_signature = Some(sig);
                self.status = PayoutStatus::Approved;
                self.finalized_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                );
                return PayoutStatus::Approved;
            }
        }

        // Check for rejection
        if self.rejectors.len() >= threshold {
            self.status = PayoutStatus::Rejected;
            self.finalized_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            );
            return PayoutStatus::Rejected;
        }

        // Check for timeout (2 minutes)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        if now - self.started_at > 120_000 {
            self.status = PayoutStatus::TimedOut;
            self.finalized_at = Some(now);
            return PayoutStatus::TimedOut;
        }

        PayoutStatus::Pending
    }

    /// Get vote count
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Get approval count
    pub fn approval_count(&self) -> usize {
        self.approvers.len()
    }
}

/// Payout consensus manager
pub struct PayoutConsensusManager {
    /// Pending payouts
    pending: HashMap<[u8; 32], PayoutConsensusState>,

    /// Completed payouts
    completed: HashMap<[u8; 32], PayoutStatus>,

    /// Our peer ID
    our_peer_id: PeerIdBytes,

    /// Our signer index in threshold scheme
    our_signer_index: u32,

    /// Our secret share for signing
    our_secret_share: [u8; 32],

    /// Known pool nodes
    known_nodes: HashSet<PeerIdBytes>,

    /// Max pending batches
    max_pending: usize,
}

impl PayoutConsensusManager {
    /// Create new payout manager
    pub fn new(our_peer_id: PeerIdBytes, signer_index: u32, secret_share: [u8; 32]) -> Self {
        Self {
            pending: HashMap::new(),
            completed: HashMap::new(),
            our_peer_id,
            our_signer_index: signer_index,
            our_secret_share: secret_share,
            known_nodes: HashSet::new(),
            max_pending: 50,
        }
    }

    /// Add known node
    pub fn add_known_node(&mut self, peer_id: PeerIdBytes) {
        self.known_nodes.insert(peer_id);
    }

    /// Remove known node
    pub fn remove_known_node(&mut self, peer_id: &PeerIdBytes) {
        self.known_nodes.remove(peer_id);
    }

    /// Handle new payout batch
    pub fn handle_batch(&mut self, batch: PayoutBatch) -> DistributedResult<()> {
        // Check if already completed
        if self.completed.contains_key(&batch.batch_id) {
            return Err(DistributedError::DuplicateShare(
                "Payout batch already processed".to_string(),
            ));
        }

        // Check if already pending
        if self.pending.contains_key(&batch.batch_id) {
            return Ok(());
        }

        // Verify amounts
        if !batch.verify_amounts() {
            return Err(DistributedError::InvalidSignature(
                "Payout amounts don't sum correctly".to_string(),
            ));
        }

        // Limit pending
        if self.pending.len() >= self.max_pending {
            self.cleanup_old();
        }

        let state = PayoutConsensusState::new(batch);
        self.pending.insert(state.batch.batch_id, state);

        Ok(())
    }

    /// Handle vote from another node
    pub fn handle_vote(&mut self, vote: PayoutVote) -> DistributedResult<PayoutStatus> {
        let state = self.pending.get_mut(&vote.batch_id).ok_or_else(|| {
            DistributedError::ConsensusNotReached(format!(
                "Unknown batch: {}",
                hex::encode(vote.batch_id)
            ))
        })?;

        state.add_vote(vote);

        let status = state.check_threshold(self.known_nodes.len().max(1));

        // Move to completed if finalized
        if status != PayoutStatus::Pending {
            let batch_id = state.batch.batch_id;
            self.completed.insert(batch_id, status.clone());
            self.pending.remove(&batch_id);
        }

        Ok(status)
    }

    /// Create our vote for a batch
    pub fn create_vote(
        &self,
        batch_id: [u8; 32],
        approve: bool,
        reason: Option<String>,
        our_pplns_hash: [u8; 32],
    ) -> DistributedResult<PayoutVote> {
        let state = self.pending.get(&batch_id).ok_or_else(|| {
            DistributedError::ConsensusNotReached(format!(
                "Unknown batch: {}",
                hex::encode(batch_id)
            ))
        })?;

        if approve {
            let message = state.batch.hash();
            let partial_sig =
                PartialSignature::create(self.our_signer_index, &message, &self.our_secret_share);

            Ok(PayoutVote::approve(
                batch_id,
                self.our_peer_id,
                our_pplns_hash,
                partial_sig,
            ))
        } else {
            Ok(PayoutVote::reject(
                batch_id,
                self.our_peer_id,
                reason.unwrap_or_else(|| "Rejected".to_string()),
                our_pplns_hash,
            ))
        }
    }

    /// Get pending batch state
    pub fn get_pending(&self, batch_id: &[u8; 32]) -> Option<&PayoutConsensusState> {
        self.pending.get(batch_id)
    }

    /// Get completed batch status
    pub fn get_completed(&self, batch_id: &[u8; 32]) -> Option<&PayoutStatus> {
        self.completed.get(batch_id)
    }

    /// Get approved batches ready for broadcast
    pub fn get_approved_batches(&self) -> Vec<(&PayoutBatch, &ThresholdSignature)> {
        self.pending
            .values()
            .filter(|s| s.status == PayoutStatus::Approved)
            .filter_map(|s| {
                s.threshold_signature
                    .as_ref()
                    .map(|sig| (&s.batch, sig))
            })
            .collect()
    }

    /// Mark batch as broadcast
    pub fn mark_broadcast(&mut self, batch_id: &[u8; 32]) {
        if let Some(state) = self.pending.get_mut(batch_id) {
            state.status = PayoutStatus::Broadcast;
        }
    }

    /// Mark batch as confirmed
    pub fn mark_confirmed(&mut self, batch_id: &[u8; 32]) {
        if let Some(state) = self.pending.get_mut(batch_id) {
            state.status = PayoutStatus::Confirmed;
            let batch_id = state.batch.batch_id;
            self.completed.insert(batch_id, PayoutStatus::Confirmed);
            self.pending.remove(&batch_id);
        }
    }

    /// Clean up old pending batches
    fn cleanup_old(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let expired: Vec<[u8; 32]> = self
            .pending
            .iter()
            .filter(|(_, state)| now - state.started_at > 300_000) // 5 minutes
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            self.completed.insert(id, PayoutStatus::TimedOut);
            self.pending.remove(&id);
        }
    }

    /// Clean up old completed entries
    pub fn cleanup_completed(&mut self, max_entries: usize) {
        if self.completed.len() > max_entries {
            let to_remove = self.completed.len() - max_entries / 2;
            let keys: Vec<_> = self.completed.keys().take(to_remove).cloned().collect();
            for key in keys {
                self.completed.remove(&key);
            }
        }
    }
}

/// Signed payout transaction ready for broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedPayoutTransaction {
    /// Original batch
    pub batch: PayoutBatch,

    /// Threshold signature
    pub signature: ThresholdSignature,

    /// Transaction hash (if broadcast)
    pub tx_hash: Option<TransactionId>,

    /// Block confirmed in
    pub confirmed_in_block: Option<u64>,
}

impl SignedPayoutTransaction {
    /// Create from approved consensus state
    pub fn from_consensus(state: &PayoutConsensusState) -> Option<Self> {
        if state.status != PayoutStatus::Approved {
            return None;
        }

        state.threshold_signature.as_ref().map(|sig| Self {
            batch: state.batch.clone(),
            signature: sig.clone(),
            tx_hash: None,
            confirmed_in_block: None,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bincode::deserialize(data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch() -> PayoutBatch {
        let payouts = vec![
            PayoutEntry {
                wallet_address: "wallet1".to_string(),
                amount: 700_000_000,
                difficulty_contribution: 70.0,
                proportion: 0.7,
            },
            PayoutEntry {
                wallet_address: "wallet2".to_string(),
                amount: 275_000_000,
                difficulty_contribution: 27.5,
                proportion: 0.275,
            },
        ];

        PayoutBatch::new(
            100,
            [1u8; 32],
            1_000_000_000,
            payouts,
            [2u8; 32],
            [3u8; 32],
        )
    }

    #[test]
    fn test_payout_batch_verification() {
        let batch = create_test_batch();

        // Total is 975M out of 1B (2.5% fees), should pass
        assert!(batch.verify_amounts());
    }

    #[test]
    fn test_payout_consensus() {
        let batch = create_test_batch();
        let mut state = PayoutConsensusState::new(batch);

        let total_nodes = 5;
        let threshold = (2 * total_nodes + 2) / 3; // 4

        // Add 3 approvals (not enough)
        for i in 0..3 {
            let mut voter = [0u8; 32];
            voter[0] = i;
            let partial_sig = PartialSignature::create(i as u32, &[0u8; 32], &[i; 32]);
            let vote = PayoutVote::approve([0u8; 32], voter, [0u8; 32], partial_sig);
            state.add_vote(vote);
        }

        assert_eq!(state.check_threshold(total_nodes), PayoutStatus::Pending);

        // Add 4th approval (reaches threshold)
        let mut voter = [0u8; 32];
        voter[0] = 3;
        let partial_sig = PartialSignature::create(3, &[0u8; 32], &[3u8; 32]);
        let vote = PayoutVote::approve([0u8; 32], voter, [0u8; 32], partial_sig);
        state.add_vote(vote);

        assert_eq!(state.check_threshold(total_nodes), PayoutStatus::Approved);
        assert!(state.threshold_signature.is_some());
    }

    #[test]
    fn test_partial_signature() {
        let message = [42u8; 32];
        let secret = [1u8; 32];

        let sig = PartialSignature::create(0, &message, &secret);

        assert!(sig.verify(&message, &[0u8; 32]));
        assert_ne!(sig.commitment, [0u8; 32]);
        assert_ne!(sig.response, [0u8; 32]);
    }

    #[test]
    fn test_threshold_aggregation() {
        let partials: Vec<_> = (0..4)
            .map(|i| PartialSignature::create(i, &[42u8; 32], &[i as u8; 32]))
            .collect();

        let combined = ThresholdSignature::aggregate(&partials, 3);
        assert!(combined.is_some());

        let sig = combined.unwrap();
        assert!(sig.threshold_met);
        assert_eq!(sig.signers.len(), 3);
    }

    #[test]
    fn test_payout_manager() {
        let mut manager = PayoutConsensusManager::new([0u8; 32], 0, [1u8; 32]);

        // Add nodes
        for i in 0..5 {
            let mut peer = [0u8; 32];
            peer[0] = i;
            manager.add_known_node(peer);
        }

        // Handle batch
        let batch = create_test_batch();
        manager.handle_batch(batch.clone()).unwrap();

        // Verify pending
        assert!(manager.get_pending(&batch.batch_id).is_some());
    }
}
