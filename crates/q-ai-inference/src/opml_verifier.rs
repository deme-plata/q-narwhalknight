//! opML Verification Protocol for Trustless AI Inference
//!
//! v6.0.0: Implements optimistic Machine Learning (opML) dispute protocol modeled
//! after Gensyn's Verde, adapted for Q-NarwhalKnight's DAG-Knight consensus.
//!
//! ## Protocol Overview
//!
//! ```text
//! 1. Worker executes inference (deterministic seed)
//! 2. Worker commits output_hash = SHA3(tokens ++ seed ++ model_hash)
//! 3. Verification lottery: SHA3(block_hash ++ request_id) mod 10 == 0? (10%)
//!    - Not selected (90%): user gets result immediately
//!    - Selected (10%): random verifier re-executes
//! 4. If hashes match: both get paid
//! 5. If mismatch: bisection dispute (log2(N) rounds)
//! 6. Arbitrator settles: loser slashed 10% of stake
//! ```
//!
//! ## Security Model
//!
//! - **Probabilistic checking**: Only 10% of requests verified (configurable)
//! - **Deterministic replay**: Same seed + model = same output (Phase 1)
//! - **Bisection dispute**: O(log n) rounds to find divergent token
//! - **Economic deterrent**: Slash >= 100 QUG per offense (10% of Bronze stake)
//! - **No trusted third party**: Arbitrator is random staked node

use crate::engine_trait::{compute_inference_commitment, DeterministicConfig, DeterministicResult};
use crate::worker_registry::WorkerRegistry;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Verification rate: 1 in N requests are randomly selected for verification
pub const VERIFICATION_RATE: u64 = 10; // 10% of requests

/// Maximum bisection rounds before forced arbitration
pub const MAX_BISECTION_ROUNDS: u32 = 20; // Supports up to 2^20 = 1M tokens

/// Verifier reward as fraction of inference fee (basis points)
pub const VERIFIER_REWARD_BPS: u64 = 500; // 5% of inference fee

/// Commitment from a worker or verifier for a specific inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCommitment {
    /// Unique request identifier
    pub request_id: [u8; 32],
    /// Address of the committing node (worker or verifier)
    pub committer: [u8; 32],
    /// SHA3-256 hash of (all_tokens ++ seed ++ model_hash)
    pub output_hash: [u8; 32],
    /// SHA3-256 prefix hashes for bisection: hash of tokens[0..i]
    pub prefix_hashes: Vec<[u8; 32]>,
    /// Number of tokens generated
    pub token_count: u32,
    /// Model hash used for inference
    pub model_hash: [u8; 32],
    /// Deterministic seed used
    pub seed: u64,
    /// Block height when committed
    pub committed_at: u64,
    /// Dilithium signature over (request_id ++ output_hash) for PQ security
    pub signature: Vec<u8>,
}

/// Phase of the dispute resolution protocol
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputePhase {
    /// Dispute just opened, waiting for both sides' commitments
    Opened,
    /// Bisection in progress: narrowing to the divergent token
    Bisecting { round: u32 },
    /// Bisection complete, arbitrator assigned to settle
    Arbitrating,
    /// Dispute resolved with winner/loser determined
    Resolved,
    /// Dispute expired (one party disappeared)
    Expired,
}

/// Resolution of a dispute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisputeResolution {
    /// Worker's output was correct; verifier made a false challenge
    WorkerCorrect {
        slashed_from_verifier: u128,
        bounty_to_worker: u128,
    },
    /// Verifier's output was correct; worker submitted fraudulent results
    VerifierCorrect {
        slashed_from_worker: u128,
        bounty_to_verifier: u128,
    },
    /// Arbitrator disagrees with both (e.g., model mismatch) — no slash
    Inconclusive { reason: String },
}

/// State of an active dispute between worker and verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeState {
    /// Request that triggered the dispute
    pub request_id: [u8; 32],
    /// Worker who executed the original inference
    pub worker: [u8; 32],
    /// Verifier who re-executed and found mismatch
    pub verifier: [u8; 32],
    /// Current dispute phase
    pub phase: DisputePhase,
    /// Worker's commitment (tokens + hashes)
    pub worker_commitment: InferenceCommitment,
    /// Verifier's commitment (tokens + hashes)
    pub verifier_commitment: InferenceCommitment,
    /// Current bisection range [lo, hi) — narrowing to divergent token
    pub bisection_lo: u32,
    pub bisection_hi: u32,
    /// Assigned arbitrator (random staked node, not worker or verifier)
    pub arbitrator: Option<[u8; 32]>,
    /// Final resolution
    pub resolution: Option<DisputeResolution>,
    /// Block height when dispute was opened
    pub opened_at: u64,
    /// Block height when dispute was resolved
    pub resolved_at: Option<u64>,
}

impl DisputeState {
    /// Create a new dispute
    pub fn new(
        request_id: [u8; 32],
        worker_commitment: InferenceCommitment,
        verifier_commitment: InferenceCommitment,
        opened_at: u64,
    ) -> Self {
        let token_count = worker_commitment.token_count;
        Self {
            request_id,
            worker: worker_commitment.committer,
            verifier: verifier_commitment.committer,
            phase: DisputePhase::Opened,
            worker_commitment,
            verifier_commitment,
            bisection_lo: 0,
            bisection_hi: token_count,
            arbitrator: None,
            resolution: None,
            opened_at,
            resolved_at: None,
        }
    }

    /// Execute one round of bisection to narrow the divergent token range.
    ///
    /// Returns the new (lo, hi) range or None if bisection is complete.
    pub fn bisect_round(&mut self) -> Option<(u32, u32)> {
        if self.bisection_hi - self.bisection_lo <= 1 {
            // Bisection complete: divergent token is at bisection_hi
            // prefix_hash[lo] matched (tokens 0..=lo identical)
            // but prefix_hash[hi] differed (token at hi is divergent)
            self.phase = DisputePhase::Arbitrating;
            return None;
        }

        let mid = (self.bisection_lo + self.bisection_hi) / 2;

        // Compare prefix hashes at midpoint
        let w_hash = self.worker_commitment.prefix_hashes.get(mid as usize);
        let v_hash = self.verifier_commitment.prefix_hashes.get(mid as usize);

        match (w_hash, v_hash) {
            (Some(w), Some(v)) => {
                if w == v {
                    // Hashes match up to mid → divergence is in [mid, hi)
                    self.bisection_lo = mid;
                } else {
                    // Hashes differ at mid → divergence is in [lo, mid)
                    self.bisection_hi = mid;
                }
            }
            _ => {
                // Missing prefix hashes → cannot bisect further
                warn!("Missing prefix hashes at index {}, forcing arbitration", mid);
                self.phase = DisputePhase::Arbitrating;
                return None;
            }
        }

        let round = match &self.phase {
            DisputePhase::Bisecting { round } => round + 1,
            _ => 1,
        };

        if round >= MAX_BISECTION_ROUNDS {
            self.phase = DisputePhase::Arbitrating;
            None
        } else {
            self.phase = DisputePhase::Bisecting { round };
            Some((self.bisection_lo, self.bisection_hi))
        }
    }

    /// Get the token index identified by bisection as the divergence point
    pub fn divergent_token_index(&self) -> u32 {
        self.bisection_hi
    }
}

/// opML Verification Manager — coordinates verification lottery, disputes, and slashing
pub struct OpMLVerifier {
    /// Active disputes: request_id -> dispute state
    disputes: Arc<RwLock<HashMap<[u8; 32], DisputeState>>>,
    /// Pending verifications: request_id -> (worker_commitment, verifier assigned)
    pending_verifications: Arc<RwLock<HashMap<[u8; 32], PendingVerification>>>,
    /// Completed dispute history (last 1000)
    dispute_history: Arc<RwLock<Vec<DisputeState>>>,
    /// Reference to worker registry for slashing
    worker_registry: Arc<WorkerRegistry>,
    /// Verification statistics
    stats: Arc<RwLock<VerificationStats>>,
}

/// Pending verification awaiting verifier result
#[derive(Debug, Clone)]
pub struct PendingVerification {
    pub worker_commitment: InferenceCommitment,
    pub verifier_address: [u8; 32],
    pub assigned_at: u64,
}

/// Verification statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationStats {
    pub total_requests: u64,
    pub total_selected_for_verification: u64,
    pub total_verified_correct: u64,
    pub total_disputes_opened: u64,
    pub total_disputes_worker_won: u64,
    pub total_disputes_verifier_won: u64,
    pub total_disputes_inconclusive: u64,
    pub total_slashed_amount: u128,
}

impl OpMLVerifier {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        info!("🔍 Initializing opML Verification Protocol");
        Self {
            disputes: Arc::new(RwLock::new(HashMap::new())),
            pending_verifications: Arc::new(RwLock::new(HashMap::new())),
            dispute_history: Arc::new(RwLock::new(Vec::new())),
            worker_registry,
            stats: Arc::new(RwLock::new(VerificationStats::default())),
        }
    }

    /// Determine if a request should be selected for verification.
    ///
    /// Selection is deterministic based on (block_hash, request_id) so that
    /// all nodes agree on which requests are verified without communication.
    pub fn should_verify(block_hash: &[u8; 32], request_id: &[u8; 32]) -> bool {
        let mut hasher = Sha3_256::new();
        hasher.update(b"opml-verification-lottery-v1");
        hasher.update(block_hash);
        hasher.update(request_id);
        let hash: [u8; 32] = hasher.finalize().into();

        // Use first 8 bytes as u64, mod VERIFICATION_RATE
        let value = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        value % VERIFICATION_RATE == 0
    }

    /// Select a random verifier from staked workers (not the original worker).
    ///
    /// Selection is deterministic based on (block_hash, request_id, "verifier")
    /// so all nodes agree on who verifies.
    pub async fn select_verifier(
        &self,
        block_hash: &[u8; 32],
        request_id: &[u8; 32],
        worker_address: &[u8; 32],
        model_hash: &[u8; 32],
    ) -> Option<[u8; 32]> {
        let eligible = self.worker_registry.get_workers_for_model(model_hash).await;
        let candidates: Vec<_> = eligible.iter()
            .filter(|w| &w.worker_address != worker_address)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Deterministic selection
        let mut hasher = Sha3_256::new();
        hasher.update(b"opml-verifier-selection-v1");
        hasher.update(block_hash);
        hasher.update(request_id);
        let hash: [u8; 32] = hasher.finalize().into();
        let index = u64::from_le_bytes(hash[0..8].try_into().unwrap()) as usize % candidates.len();

        Some(candidates[index].worker_address)
    }

    /// Select a random arbitrator (not worker, not verifier)
    pub async fn select_arbitrator(
        &self,
        block_hash: &[u8; 32],
        request_id: &[u8; 32],
        worker: &[u8; 32],
        verifier: &[u8; 32],
        model_hash: &[u8; 32],
    ) -> Option<[u8; 32]> {
        let eligible = self.worker_registry.get_workers_for_model(model_hash).await;
        let candidates: Vec<_> = eligible.iter()
            .filter(|w| &w.worker_address != worker && &w.worker_address != verifier)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let mut hasher = Sha3_256::new();
        hasher.update(b"opml-arbitrator-selection-v1");
        hasher.update(block_hash);
        hasher.update(request_id);
        let hash: [u8; 32] = hasher.finalize().into();
        let index = u64::from_le_bytes(hash[0..8].try_into().unwrap()) as usize % candidates.len();

        Some(candidates[index].worker_address)
    }

    /// Submit worker's inference commitment (called after inference completes)
    pub async fn submit_worker_commitment(
        &self,
        commitment: InferenceCommitment,
        block_hash: &[u8; 32],
        current_height: u64,
    ) -> Result<VerificationAction> {
        let request_id = commitment.request_id;
        self.stats.write().await.total_requests += 1;

        // Check if this request is selected for verification
        if !Self::should_verify(block_hash, &request_id) {
            return Ok(VerificationAction::NotSelected);
        }

        self.stats.write().await.total_selected_for_verification += 1;

        // Select verifier
        let verifier = self.select_verifier(
            block_hash,
            &request_id,
            &commitment.committer,
            &commitment.model_hash,
        ).await;

        match verifier {
            Some(verifier_address) => {
                self.pending_verifications.write().await.insert(
                    request_id,
                    PendingVerification {
                        worker_commitment: commitment,
                        verifier_address,
                        assigned_at: current_height,
                    },
                );

                Ok(VerificationAction::VerificationRequested {
                    verifier: verifier_address,
                })
            }
            None => {
                // No eligible verifier available — pass without verification
                debug!("No eligible verifier for request, passing without verification");
                Ok(VerificationAction::NotSelected)
            }
        }
    }

    /// Submit verifier's commitment (re-execution result)
    pub async fn submit_verifier_commitment(
        &self,
        commitment: InferenceCommitment,
        block_hash: &[u8; 32],
        current_height: u64,
    ) -> Result<VerificationAction> {
        let request_id = commitment.request_id;

        let pending = self.pending_verifications.write().await.remove(&request_id);
        let pending = match pending {
            Some(p) => p,
            None => return Err(anyhow!("No pending verification for request")),
        };

        // Check: is the committer the assigned verifier?
        if commitment.committer != pending.verifier_address {
            return Err(anyhow!("Commitment from wrong verifier"));
        }

        // Compare output hashes
        if pending.worker_commitment.output_hash == commitment.output_hash {
            // Match! Both correct.
            self.stats.write().await.total_verified_correct += 1;
            self.worker_registry.record_challenge_passed(pending.worker_commitment.committer).await;

            Ok(VerificationAction::Verified {
                worker: pending.worker_commitment.committer,
                verifier: commitment.committer,
            })
        } else {
            // Mismatch! Open dispute.
            let dispute = DisputeState::new(
                request_id,
                pending.worker_commitment,
                commitment,
                current_height,
            );

            self.stats.write().await.total_disputes_opened += 1;
            self.disputes.write().await.insert(request_id, dispute);

            info!("⚠️ opML dispute opened for request {:?}", hex::encode(request_id));

            Ok(VerificationAction::DisputeOpened { request_id })
        }
    }

    /// Execute bisection round for an active dispute
    pub async fn bisect(&self, request_id: &[u8; 32]) -> Result<BisectionResult> {
        let mut disputes = self.disputes.write().await;
        let dispute = disputes.get_mut(request_id)
            .ok_or_else(|| anyhow!("No active dispute for request"))?;

        match dispute.bisect_round() {
            Some((lo, hi)) => Ok(BisectionResult::Continue { lo, hi }),
            None => {
                let divergent = dispute.divergent_token_index();
                Ok(BisectionResult::Complete { divergent_token: divergent })
            }
        }
    }

    /// Submit arbitrator's result and resolve the dispute
    pub async fn resolve_dispute(
        &self,
        request_id: &[u8; 32],
        arbitrator_hash: [u8; 32],
        current_height: u64,
    ) -> Result<DisputeResolution> {
        let mut disputes = self.disputes.write().await;
        let dispute = disputes.get_mut(request_id)
            .ok_or_else(|| anyhow!("No active dispute for request"))?;

        // Compare arbitrator's hash with worker and verifier
        let worker_matches = dispute.worker_commitment.output_hash == arbitrator_hash;
        let verifier_matches = dispute.verifier_commitment.output_hash == arbitrator_hash;

        let resolution = match (worker_matches, verifier_matches) {
            (true, false) => {
                // Worker correct, verifier wrong
                let slashed = self.worker_registry.slash_worker(
                    dispute.verifier, "False opML challenge"
                ).await.unwrap_or(0);

                self.stats.write().await.total_disputes_worker_won += 1;
                self.stats.write().await.total_slashed_amount += slashed;

                DisputeResolution::WorkerCorrect {
                    slashed_from_verifier: slashed,
                    bounty_to_worker: slashed / 2, // Half goes to worker as bounty
                }
            }
            (false, true) => {
                // Verifier correct, worker cheated
                let slashed = self.worker_registry.slash_worker(
                    dispute.worker, "Fraudulent inference output"
                ).await.unwrap_or(0);

                self.stats.write().await.total_disputes_verifier_won += 1;
                self.stats.write().await.total_slashed_amount += slashed;

                DisputeResolution::VerifierCorrect {
                    slashed_from_worker: slashed,
                    bounty_to_verifier: slashed / 2,
                }
            }
            _ => {
                // Both wrong or both right — inconclusive (model hash mismatch?)
                self.stats.write().await.total_disputes_inconclusive += 1;
                DisputeResolution::Inconclusive {
                    reason: "Arbitrator disagrees with both parties".into(),
                }
            }
        };

        dispute.resolution = Some(resolution.clone());
        dispute.resolved_at = Some(current_height);
        dispute.phase = DisputePhase::Resolved;

        // Move to history
        let resolved = disputes.remove(request_id).unwrap();
        let mut history = self.dispute_history.write().await;
        history.push(resolved);
        if history.len() > 1000 {
            let excess = history.len() - 1000;
            history.drain(0..excess);
        }

        Ok(resolution)
    }

    /// Get verification statistics
    pub async fn get_stats(&self) -> VerificationStats {
        self.stats.read().await.clone()
    }

    /// Get active disputes count
    pub async fn active_disputes(&self) -> usize {
        self.disputes.read().await.len()
    }

    /// Expire old pending verifications (verifier didn't respond within 100 blocks)
    pub async fn expire_stale(&self, current_height: u64) {
        let mut pending = self.pending_verifications.write().await;
        pending.retain(|_, v| current_height.saturating_sub(v.assigned_at) < 100);

        let mut disputes = self.disputes.write().await;
        let expired: Vec<[u8; 32]> = disputes.iter()
            .filter(|(_, d)| {
                d.phase != DisputePhase::Resolved
                    && current_height.saturating_sub(d.opened_at) > 200
            })
            .map(|(id, _)| *id)
            .collect();

        for id in expired {
            if let Some(mut d) = disputes.remove(&id) {
                d.phase = DisputePhase::Expired;
                d.resolved_at = Some(current_height);
                self.dispute_history.write().await.push(d);
            }
        }
    }
}

/// What action to take after submitting a commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationAction {
    /// Request not selected for verification — deliver result immediately
    NotSelected,
    /// Verification requested — assigned verifier must re-execute
    VerificationRequested { verifier: [u8; 32] },
    /// Verification passed — both worker and verifier produced same result
    Verified { worker: [u8; 32], verifier: [u8; 32] },
    /// Dispute opened — worker and verifier disagree
    DisputeOpened { request_id: [u8; 32] },
}

/// Result of a bisection round
#[derive(Debug, Clone)]
pub enum BisectionResult {
    /// Bisection continues with narrowed range
    Continue { lo: u32, hi: u32 },
    /// Bisection complete — divergent token identified
    Complete { divergent_token: u32 },
}

/// Gossipsub message types for the opML verification protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpMLMessage {
    /// Worker submits inference commitment
    WorkerCommitment(InferenceCommitment),
    /// Verifier assigned to re-execute
    VerificationChallenge {
        request_id: [u8; 32],
        verifier: [u8; 32],
        worker_commitment: InferenceCommitment,
    },
    /// Verifier submits re-execution result
    VerificationResult {
        request_id: [u8; 32],
        verifier_commitment: InferenceCommitment,
    },
    /// Dispute opened (hashes differ)
    DisputeOpen {
        request_id: [u8; 32],
        worker: [u8; 32],
        verifier: [u8; 32],
    },
    /// Bisection round result
    DisputeBisect {
        request_id: [u8; 32],
        round: u32,
        range_lo: u32,
        range_hi: u32,
    },
    /// Arbitrator assigned
    DisputeArbitrate {
        request_id: [u8; 32],
        arbitrator: [u8; 32],
        divergent_token: u32,
    },
    /// Dispute resolved
    DisputeResolve {
        request_id: [u8; 32],
        resolution: DisputeResolution,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_lottery_deterministic() {
        let block_hash = [1u8; 32];
        let request_id = [2u8; 32];

        // Same inputs always produce same result
        let result1 = OpMLVerifier::should_verify(&block_hash, &request_id);
        let result2 = OpMLVerifier::should_verify(&block_hash, &request_id);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_verification_lottery_distribution() {
        let block_hash = [42u8; 32];
        let mut selected = 0u64;
        let total = 10_000u64;

        for i in 0..total {
            let mut request_id = [0u8; 32];
            request_id[0..8].copy_from_slice(&i.to_le_bytes());
            if OpMLVerifier::should_verify(&block_hash, &request_id) {
                selected += 1;
            }
        }

        // Should be approximately 10% (within 3% tolerance)
        let rate = selected as f64 / total as f64;
        assert!(rate > 0.07 && rate < 0.13,
            "Verification rate {:.2}% outside expected range 7-13%", rate * 100.0);
    }

    #[test]
    fn test_bisection() {
        let worker_tokens = vec!["hello".to_string(), "world".to_string(), "foo".to_string(), "bar".to_string()];
        let verifier_tokens = vec!["hello".to_string(), "world".to_string(), "baz".to_string(), "bar".to_string()];
        // Divergence at index 2

        let seed = 42u64;
        let model_hash = [0u8; 32];

        let (w_hash, w_prefixes) = compute_inference_commitment(&worker_tokens, seed, &model_hash);
        let (v_hash, v_prefixes) = compute_inference_commitment(&verifier_tokens, seed, &model_hash);

        assert_ne!(w_hash, v_hash);

        let w_commit = InferenceCommitment {
            request_id: [1u8; 32],
            committer: [10u8; 32],
            output_hash: w_hash,
            prefix_hashes: w_prefixes,
            token_count: 4,
            model_hash,
            seed,
            committed_at: 100,
            signature: vec![],
        };

        let v_commit = InferenceCommitment {
            request_id: [1u8; 32],
            committer: [20u8; 32],
            output_hash: v_hash,
            prefix_hashes: v_prefixes,
            token_count: 4,
            model_hash,
            seed,
            committed_at: 100,
            signature: vec![],
        };

        let mut dispute = DisputeState::new([1u8; 32], w_commit, v_commit, 100);

        // Bisect until we find the divergent token
        loop {
            match dispute.bisect_round() {
                Some(_range) => continue,
                None => break,
            }
        }

        assert_eq!(dispute.divergent_token_index(), 2, "Should find divergence at token index 2");
    }
}
