//! P2P Message Types for Recursive Proof Protocol
//!
//! Defines the message formats for proof task announcements,
//! proof submissions, and light client interactions.

use crate::{EpochProof, EpochPublicInputs};
use q_lattice_guard::LatticeGuardProof;
use serde::{Deserialize, Serialize};

/// Epoch proof task announcement
///
/// Broadcast when a new epoch is finalized and needs a recursive proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochProofTask {
    /// Epoch number to prove
    pub epoch: u64,

    /// Block height range covered
    pub height_start: u64,
    pub height_end: u64,

    /// Previous epoch's proof hash (for verification)
    pub previous_proof_hash: [u8; 32],

    /// Previous state root
    pub previous_state_root: [u8; 32],

    /// Current state root (to be proven)
    pub current_state_root: [u8; 32],

    /// Validator set hash for this epoch
    pub validator_set_hash: [u8; 32],

    /// Block hashes in this epoch (for provers to fetch full data)
    pub block_hashes: Vec<[u8; 32]>,

    /// BFT signature references (provers fetch actual signatures)
    pub signature_refs: Vec<SignatureRef>,

    /// Deadline for proof submission (Unix timestamp)
    pub deadline: u64,

    /// Reward offered for proof generation (in smallest unit)
    pub reward: u64,

    /// Network ID
    pub network_id: String,

    /// Task issuer (validator that finalized the epoch)
    pub issuer: String,
}

/// Reference to a BFT signature
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureRef {
    /// Validator public key hash
    pub validator_hash: [u8; 32],
    /// Signature hash (for verification)
    pub signature_hash: [u8; 32],
}

/// Epoch proof submission
///
/// Submitted by provers when they complete a proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochProofSubmission {
    /// Epoch this proof covers
    pub epoch: u64,

    /// The recursive SNARK proof (serialized)
    pub proof_data: Vec<u8>,

    /// Public inputs (serialized)
    pub public_inputs_data: Vec<u8>,

    /// Prover's peer ID
    pub prover_peer_id: String,

    /// Prover's signature on the proof (proves ownership)
    pub prover_signature: Vec<u8>,

    /// Proving time in milliseconds
    pub proving_time_ms: u64,

    /// Hardware info (optional, for benchmarking)
    pub hardware_info: Option<HardwareInfo>,

    /// Protocol version
    pub protocol_version: u32,

    /// Timestamp when proof was created
    pub created_at: u64,
}

impl EpochProofSubmission {
    /// Create from epoch proof
    pub fn from_proof(
        epoch: u64,
        proof: &EpochProof,
        prover_peer_id: String,
        proving_time_ms: u64,
    ) -> Result<Self, bincode::Error> {
        Ok(Self {
            epoch,
            proof_data: bincode::serialize(&proof.proof)?,
            public_inputs_data: bincode::serialize(&proof.public_inputs)?,
            prover_peer_id,
            prover_signature: Vec::new(), // To be signed
            proving_time_ms,
            hardware_info: None,
            protocol_version: 1,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Deserialize to epoch proof
    pub fn to_proof(&self) -> Result<(LatticeGuardProof, EpochPublicInputs), bincode::Error> {
        let proof: LatticeGuardProof = bincode::deserialize(&self.proof_data)?;
        let public_inputs: EpochPublicInputs = bincode::deserialize(&self.public_inputs_data)?;
        Ok((proof, public_inputs))
    }

    /// Compute submission hash (for signing/verification)
    pub fn hash(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(&self.proof_data);
        hasher.update(&self.public_inputs_data);
        hasher.update(self.prover_peer_id.as_bytes());

        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }
}

/// Hardware information for benchmarking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model
    pub cpu_model: Option<String>,
    /// Number of CPU cores
    pub cpu_cores: Option<u32>,
    /// GPU model (if used)
    pub gpu_model: Option<String>,
    /// RAM in GB
    pub ram_gb: Option<u32>,
    /// Used GPU acceleration
    pub used_gpu: bool,
}

/// Proof verification result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofVerificationResult {
    /// Epoch this result is for
    pub epoch: u64,
    /// Prover peer ID
    pub prover_peer_id: String,
    /// Is the proof valid?
    pub is_valid: bool,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
    /// Error message (if invalid)
    pub error_message: Option<String>,
    /// Verifier peer ID
    pub verifier_peer_id: String,
}

/// Light client proof request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightClientProofRequest {
    /// Requester's current known height (0 for new nodes)
    pub known_height: u64,

    /// Requester's current known epoch (0 for new nodes)
    pub known_epoch: u64,

    /// Whether to include full validator set
    pub include_validators: bool,

    /// Requester peer ID
    pub requester_peer_id: String,

    /// Request timestamp
    pub timestamp: u64,
}

/// Light client proof response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightClientProofResponse {
    /// The accumulated proof covering all history (serialized)
    pub proof_data: Vec<u8>,

    /// Current state root
    pub current_state_root: [u8; 32],

    /// Current height
    pub current_height: u64,

    /// Current epoch
    pub current_epoch: u64,

    /// Validator set (if requested)
    pub validator_set: Option<ValidatorSet>,

    /// Proof of validator set correctness
    pub validator_set_proof: Option<Vec<u8>>,

    /// Response timestamp
    pub timestamp: u64,

    /// Responder peer ID
    pub responder_peer_id: String,
}

/// Validator information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Validator public key (Dilithium)
    pub public_key: Vec<u8>,
    /// Validator's stake amount
    pub stake: u64,
    /// Validator index
    pub index: u32,
    /// Validator peer ID (for P2P)
    pub peer_id: Option<String>,
}

/// Complete validator set
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorSet {
    /// All validators
    pub validators: Vec<ValidatorInfo>,
    /// Total stake
    pub total_stake: u64,
    /// Epoch this set is valid for
    pub epoch: u64,
    /// Set hash
    pub set_hash: [u8; 32],
}

/// Prover reward claim
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverRewardClaim {
    /// Epoch the reward is for
    pub epoch: u64,
    /// Prover peer ID
    pub prover_peer_id: String,
    /// Proof hash
    pub proof_hash: [u8; 32],
    /// Claimed reward amount
    pub reward_amount: u64,
    /// Prover's signature on the claim
    pub claim_signature: Vec<u8>,
}

/// Reward calculation parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewardParams {
    /// Base reward for generating an epoch proof
    pub base_reward: u64,
    /// Speed bonus multiplier (for fast proofs)
    pub speed_bonus_multiplier: f64,
    /// Target proving time in milliseconds
    pub target_proving_time_ms: u64,
    /// Late penalty per second (after deadline)
    pub late_penalty_per_second: u64,
}

impl Default for RewardParams {
    fn default() -> Self {
        Self {
            base_reward: 1_000_000,              // 1 QNK
            speed_bonus_multiplier: 0.5,         // Up to 50% bonus
            target_proving_time_ms: 10_000,      // 10 seconds target
            late_penalty_per_second: 10_000,     // 0.01 QNK per second late
        }
    }
}

impl RewardParams {
    /// Calculate reward for a proof submission
    pub fn calculate_reward(&self, submission: &EpochProofSubmission, task: &EpochProofTask) -> u64 {
        let mut reward = self.base_reward;

        // Speed bonus for fast proofs
        if submission.proving_time_ms < self.target_proving_time_ms {
            let speedup_ratio = 1.0 - (submission.proving_time_ms as f64 / self.target_proving_time_ms as f64);
            let bonus = (self.base_reward as f64 * self.speed_bonus_multiplier * speedup_ratio) as u64;
            reward += bonus;
        }

        // Late penalty (if after deadline)
        if submission.created_at > task.deadline {
            let late_seconds = submission.created_at - task.deadline;
            let penalty = self.late_penalty_per_second * late_seconds;
            reward = reward.saturating_sub(penalty);
        }

        reward
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_proof_task_serialization() {
        let task = EpochProofTask {
            epoch: 42,
            height_start: 1000,
            height_end: 2000,
            previous_proof_hash: [1u8; 32],
            previous_state_root: [2u8; 32],
            current_state_root: [3u8; 32],
            validator_set_hash: [4u8; 32],
            block_hashes: vec![[5u8; 32], [6u8; 32]],
            signature_refs: vec![],
            deadline: 1700000000,
            reward: 1_000_000,
            network_id: "testnet".to_string(),
            issuer: "validator1".to_string(),
        };

        let serialized = bincode::serialize(&task).unwrap();
        let deserialized: EpochProofTask = bincode::deserialize(&serialized).unwrap();

        assert_eq!(task.epoch, deserialized.epoch);
        assert_eq!(task.height_start, deserialized.height_start);
    }

    #[test]
    fn test_reward_calculation() {
        let params = RewardParams::default();

        let task = EpochProofTask {
            epoch: 1,
            deadline: 1700000000,
            ..Default::default()
        };

        // Fast proof
        let fast_submission = EpochProofSubmission {
            epoch: 1,
            proof_data: vec![],
            public_inputs_data: vec![],
            prover_peer_id: "prover1".to_string(),
            prover_signature: vec![],
            proving_time_ms: 5000, // 5 seconds (half of target)
            hardware_info: None,
            protocol_version: 1,
            created_at: 1699999990, // Before deadline
        };

        let fast_reward = params.calculate_reward(&fast_submission, &task);
        println!("Fast proof reward: {}", fast_reward);

        // Should get speed bonus
        assert!(fast_reward > params.base_reward);

        // Late proof
        let late_submission = EpochProofSubmission {
            created_at: 1700000010, // 10 seconds late
            ..fast_submission.clone()
        };

        let late_reward = params.calculate_reward(&late_submission, &task);
        println!("Late proof reward: {}", late_reward);

        // Should have penalty
        assert!(late_reward < fast_reward);
    }
}

impl Default for EpochProofTask {
    fn default() -> Self {
        Self {
            epoch: 0,
            height_start: 0,
            height_end: 0,
            previous_proof_hash: [0u8; 32],
            previous_state_root: [0u8; 32],
            current_state_root: [0u8; 32],
            validator_set_hash: [0u8; 32],
            block_hashes: vec![],
            signature_refs: vec![],
            deadline: 0,
            reward: 0,
            network_id: String::new(),
            issuer: String::new(),
        }
    }
}
