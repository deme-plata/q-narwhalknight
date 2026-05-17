/// v1.4.11-beta: Commit-Reveal Mining Protection
/// v2.4.1-beta: TemporalShield-enhanced commits (2-of-3 threshold, HNDL-resistant)
///
/// Provides cryptographic time-locks for mining submissions to prevent:
/// - Front-running attacks (seeing nonce before reveal)
/// - MEV (Miner Extractable Value) theft
/// - Block withholding with instant reveal
/// - HNDL (Harvest Now, Decrypt Later) attacks (via TemporalShield)
///
/// Flow (Standard):
/// 1. Miner commits: H(nonce || miner_address || block_height || secret)
/// 2. Wait REVEAL_DELAY blocks (2-5 blocks)
/// 3. Miner reveals: (nonce, secret) - verified against commitment
///
/// Flow (Temporal-Protected - 2-of-3 threshold):
/// 1. Miner creates TemporalMiningCommit with protected reveal data
/// 2. Trustees release shares according to staggered schedule:
///    - Trustee 0: height + 2
///    - Trustee 1: height + 4
///    - Trustee 2: height + 6
/// 3. At height + 4 (when 2 shares available), reveal can be reconstructed
///
/// Security Properties:
/// - Binding: Cannot change nonce after commitment
/// - Hiding: Cannot determine nonce from commitment (secret provides entropy)
/// - Time-locked: Must wait before revealing (prevents front-running)
/// - Post-quantum: ML-KEM-1024 encrypted shares (HNDL-resistant)
/// - Information-theoretic: OTP encryption with threshold shares

use blake3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Minimum blocks before reveal is accepted
pub const MIN_REVEAL_DELAY: u64 = 2;
/// Maximum blocks before commitment expires
pub const MAX_REVEAL_DELAY: u64 = 10;

/// Temporal commit-reveal threshold parameters (2-of-3)
pub const TEMPORAL_COMMIT_THRESHOLD: usize = 2;
pub const TEMPORAL_COMMIT_TOTAL_TRUSTEES: usize = 3;

/// Staggered reveal schedule offsets (blocks from commit height)
pub const TRUSTEE_0_REVEAL_OFFSET: u64 = 2;
pub const TRUSTEE_1_REVEAL_OFFSET: u64 = 4;
pub const TRUSTEE_2_REVEAL_OFFSET: u64 = 6;

/// v2.4.1-beta: TemporalShield-protected mining commitment
///
/// Uses (2,3) threshold secret sharing with post-quantum encryption.
/// Trustees release shares according to a staggered schedule, preventing
/// early reveal while ensuring eventual data availability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMiningCommit {
    /// Blake3 hash of (nonce || miner_address || block_height)
    pub commitment_hash: [u8; 32],
    /// TemporalEnvelope containing protected reveal data (nonce, secret)
    /// Encoded bytes from TemporalEnvelope::to_bytes()
    pub protected_reveal: Vec<u8>,
    /// Miner's wallet address
    pub miner_address: String,
    /// Block height when commitment was made
    pub block_height: u64,
    /// Staggered reveal schedule: (block_height, trustee_index)
    /// Each entry indicates when a trustee should release their share
    pub reveal_schedule: Vec<(u64, usize)>,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Number of shares released so far
    pub shares_released: usize,
    /// Whether reconstruction is possible (shares_released >= threshold)
    pub can_reconstruct: bool,
    /// STARK proof for commitment integrity (NO TRUSTED SETUP)
    pub stark_proof: Vec<u8>,
    /// Key commitment for verification
    pub key_commitment: [u8; 32],
}

impl TemporalMiningCommit {
    /// Create new temporal mining commit with staggered reveal schedule
    pub fn new(
        commitment_hash: [u8; 32],
        protected_reveal: Vec<u8>,
        miner_address: String,
        block_height: u64,
        stark_proof: Vec<u8>,
        key_commitment: [u8; 32],
    ) -> Self {
        let reveal_schedule = vec![
            (block_height + TRUSTEE_0_REVEAL_OFFSET, 0),
            (block_height + TRUSTEE_1_REVEAL_OFFSET, 1),
            (block_height + TRUSTEE_2_REVEAL_OFFSET, 2),
        ];

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            commitment_hash,
            protected_reveal,
            miner_address,
            block_height,
            reveal_schedule,
            created_at: timestamp,
            shares_released: 0,
            can_reconstruct: false,
            stark_proof,
            key_commitment,
        }
    }

    /// Check if a trustee should release their share at given height
    pub fn should_release_share(&self, trustee_index: usize, current_height: u64) -> bool {
        self.reveal_schedule
            .iter()
            .find(|(_, idx)| *idx == trustee_index)
            .map(|(height, _)| current_height >= *height)
            .unwrap_or(false)
    }

    /// Record a share release and update reconstruction status
    pub fn record_share_release(&mut self) {
        self.shares_released += 1;
        if self.shares_released >= TEMPORAL_COMMIT_THRESHOLD {
            self.can_reconstruct = true;
        }
    }

    /// Get next scheduled release (block_height, trustee_index)
    pub fn next_scheduled_release(&self, current_height: u64) -> Option<(u64, usize)> {
        self.reveal_schedule
            .iter()
            .filter(|(height, _)| *height > current_height)
            .min_by_key(|(height, _)| *height)
            .copied()
    }

    /// Earliest height at which reconstruction becomes possible
    pub fn earliest_reconstruction_height(&self) -> u64 {
        // With (2,3) threshold, reconstruction is possible when 2nd share releases
        self.block_height + TRUSTEE_1_REVEAL_OFFSET
    }

    /// Serialize to bytes for storage
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes)
            .map_err(|e| format!("Deserialization failed: {}", e))
    }

    /// Get commitment hash as hex string
    pub fn commitment_hex(&self) -> String {
        hex::encode(self.commitment_hash)
    }
}

/// Reveal data structure for temporal commits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRevealData {
    /// The mining nonce
    pub nonce: u64,
    /// Secret used in commitment
    pub secret: [u8; 32],
}

impl TemporalRevealData {
    pub fn new(nonce: u64, secret: [u8; 32]) -> Self {
        Self { nonce, secret }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(40);
        bytes.extend_from_slice(&self.nonce.to_le_bytes());
        bytes.extend_from_slice(&self.secret);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 40 {
            return Err("Invalid reveal data length".to_string());
        }
        let nonce = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&bytes[8..40]);
        Ok(Self { nonce, secret })
    }
}

/// Status of temporal commit shares
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalShareStatus {
    /// Trustee index
    pub trustee_index: usize,
    /// Block height at which share can be released
    pub release_height: u64,
    /// Whether share has been released
    pub released: bool,
    /// Release timestamp (0 if not released)
    pub released_at: u64,
}

/// Commitment status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitmentStatus {
    /// Commitment accepted, waiting for reveal
    Pending,
    /// Commitment revealed and verified
    Revealed,
    /// Commitment expired (not revealed in time)
    Expired,
    /// Commitment failed verification on reveal
    Invalid,
}

/// Mining commitment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningCommitment {
    /// Blake3 hash of (nonce || miner_address || block_height || secret)
    pub commitment_hash: [u8; 32],
    /// Miner's wallet address
    pub miner_address: String,
    /// Block height when commitment was made
    pub commit_height: u64,
    /// Earliest block height for reveal (commit_height + MIN_REVEAL_DELAY)
    pub reveal_after: u64,
    /// Expiry block height (commit_height + MAX_REVEAL_DELAY)
    pub expires_at: u64,
    /// Unix timestamp of commitment
    pub timestamp: u64,
    /// Whether this commitment has been revealed
    pub revealed: bool,
    /// Revealed nonce (only set after successful reveal)
    pub revealed_nonce: Option<u64>,
}

/// Reveal request from miner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningReveal {
    /// The nonce being revealed
    pub nonce: u64,
    /// Secret used in commitment (32 bytes, hex encoded)
    pub secret: String,
    /// Block height this reveal is for
    pub block_height: u64,
    /// Original commitment hash (to identify the commitment)
    pub commitment_hash: String,
}

/// Commit request from miner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningCommitRequest {
    /// Blake3 hash of (nonce || miner_address || block_height || secret)
    /// Hex encoded, 64 characters
    pub commitment_hash: String,
    /// Miner's wallet address (qnk...)
    pub miner_address: String,
    /// Current block height (commitment is tied to this height)
    pub block_height: u64,
}

/// Commit-Reveal Mining Manager
pub struct CommitRevealManager {
    /// Active commitments: commitment_hash -> MiningCommitment
    commitments: Arc<RwLock<HashMap<[u8; 32], MiningCommitment>>>,
    /// Commitments by miner: miner_address -> Vec<commitment_hash>
    miner_commitments: Arc<RwLock<HashMap<String, Vec<[u8; 32]>>>>,
    /// Enable/disable commit-reveal (for gradual rollout)
    enabled: bool,
}

impl Default for CommitRevealManager {
    fn default() -> Self {
        Self::new(true)
    }
}

impl CommitRevealManager {
    /// Create new manager
    pub fn new(enabled: bool) -> Self {
        Self {
            commitments: Arc::new(RwLock::new(HashMap::new())),
            miner_commitments: Arc::new(RwLock::new(HashMap::new())),
            enabled,
        }
    }

    /// Check if commit-reveal is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register a new commitment
    pub async fn register_commitment(
        &self,
        request: MiningCommitRequest,
    ) -> Result<MiningCommitment, String> {
        // Parse commitment hash
        let commit_bytes = hex::decode(&request.commitment_hash)
            .map_err(|e| format!("Invalid commitment hash hex: {}", e))?;

        if commit_bytes.len() != 32 {
            return Err("Commitment hash must be 32 bytes".to_string());
        }

        let mut commitment_hash = [0u8; 32];
        commitment_hash.copy_from_slice(&commit_bytes);

        // Validate miner address
        if !request.miner_address.starts_with("qnk") || request.miner_address.len() != 67 {
            return Err("Invalid miner address format".to_string());
        }

        // Create commitment record
        let commitment = MiningCommitment {
            commitment_hash,
            miner_address: request.miner_address.clone(),
            commit_height: request.block_height,
            reveal_after: request.block_height + MIN_REVEAL_DELAY,
            expires_at: request.block_height + MAX_REVEAL_DELAY,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            revealed: false,
            revealed_nonce: None,
        };

        // Store commitment
        {
            let mut commitments = self.commitments.write().await;
            if commitments.contains_key(&commitment_hash) {
                return Err("Commitment already exists".to_string());
            }
            commitments.insert(commitment_hash, commitment.clone());
        }

        // Track by miner
        {
            let mut miner_commits = self.miner_commitments.write().await;
            miner_commits
                .entry(request.miner_address.clone())
                .or_insert_with(Vec::new)
                .push(commitment_hash);
        }

        info!(
            "🔒 [COMMIT-REVEAL] Registered commitment: {} from {} at height {}",
            &request.commitment_hash[..16],
            &request.miner_address[..16],
            request.block_height
        );

        Ok(commitment)
    }

    /// Verify and process a reveal
    pub async fn process_reveal(
        &self,
        reveal: MiningReveal,
        miner_address: &str,
        current_height: u64,
    ) -> Result<u64, String> {
        // Parse commitment hash
        let commit_bytes = hex::decode(&reveal.commitment_hash)
            .map_err(|e| format!("Invalid commitment hash hex: {}", e))?;

        if commit_bytes.len() != 32 {
            return Err("Commitment hash must be 32 bytes".to_string());
        }

        let mut commitment_hash = [0u8; 32];
        commitment_hash.copy_from_slice(&commit_bytes);

        // Parse secret
        let secret_bytes = hex::decode(&reveal.secret)
            .map_err(|e| format!("Invalid secret hex: {}", e))?;

        if secret_bytes.len() != 32 {
            return Err("Secret must be 32 bytes".to_string());
        }

        // Get commitment
        let mut commitments = self.commitments.write().await;
        let commitment = commitments
            .get_mut(&commitment_hash)
            .ok_or("Commitment not found")?;

        // Verify miner matches
        if commitment.miner_address != miner_address {
            return Err("Miner address mismatch".to_string());
        }

        // Check if already revealed
        if commitment.revealed {
            return Err("Commitment already revealed".to_string());
        }

        // Check timing constraints
        if current_height < commitment.reveal_after {
            return Err(format!(
                "Too early to reveal. Wait until height {} (current: {})",
                commitment.reveal_after, current_height
            ));
        }

        if current_height > commitment.expires_at {
            return Err(format!(
                "Commitment expired at height {} (current: {})",
                commitment.expires_at, current_height
            ));
        }

        // Verify the reveal matches the commitment
        // commitment_hash = Blake3(nonce || miner_address || block_height || secret)
        let mut preimage = Vec::new();
        preimage.extend_from_slice(&reveal.nonce.to_le_bytes());
        preimage.extend_from_slice(miner_address.as_bytes());
        preimage.extend_from_slice(&reveal.block_height.to_le_bytes());
        preimage.extend_from_slice(&secret_bytes);

        let computed_hash = blake3::hash(&preimage);

        if computed_hash.as_bytes() != &commitment_hash {
            warn!(
                "❌ [COMMIT-REVEAL] Hash mismatch for {} - invalid reveal",
                &miner_address[..16]
            );
            return Err("Reveal does not match commitment".to_string());
        }

        // Mark as revealed
        commitment.revealed = true;
        commitment.revealed_nonce = Some(reveal.nonce);

        info!(
            "✅ [COMMIT-REVEAL] Valid reveal from {} - nonce: {} at height {}",
            &miner_address[..16],
            reveal.nonce,
            current_height
        );

        Ok(reveal.nonce)
    }

    /// Generate commitment hash (helper for miners)
    pub fn compute_commitment(
        nonce: u64,
        miner_address: &str,
        block_height: u64,
        secret: &[u8; 32],
    ) -> [u8; 32] {
        let mut preimage = Vec::new();
        preimage.extend_from_slice(&nonce.to_le_bytes());
        preimage.extend_from_slice(miner_address.as_bytes());
        preimage.extend_from_slice(&block_height.to_le_bytes());
        preimage.extend_from_slice(secret);

        *blake3::hash(&preimage).as_bytes()
    }

    /// Cleanup expired commitments
    pub async fn cleanup_expired(&self, current_height: u64) {
        let mut to_remove = Vec::new();

        {
            let commitments = self.commitments.read().await;
            for (hash, commitment) in commitments.iter() {
                if current_height > commitment.expires_at && !commitment.revealed {
                    to_remove.push(*hash);
                }
            }
        }

        if !to_remove.is_empty() {
            let mut commitments = self.commitments.write().await;
            for hash in &to_remove {
                if let Some(c) = commitments.remove(hash) {
                    debug!(
                        "🗑️ [COMMIT-REVEAL] Expired commitment from {} (height {})",
                        &c.miner_address[..16],
                        c.commit_height
                    );
                }
            }
            info!(
                "🧹 [COMMIT-REVEAL] Cleaned up {} expired commitments",
                to_remove.len()
            );
        }
    }

    /// Get pending commitments for a miner
    pub async fn get_miner_commitments(&self, miner_address: &str) -> Vec<MiningCommitment> {
        let miner_commits = self.miner_commitments.read().await;
        let commitments = self.commitments.read().await;

        miner_commits
            .get(miner_address)
            .map(|hashes| {
                hashes
                    .iter()
                    .filter_map(|h| commitments.get(h).cloned())
                    .filter(|c| !c.revealed)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> CommitRevealStats {
        let commitments = self.commitments.read().await;

        let total = commitments.len();
        let pending = commitments.values().filter(|c| !c.revealed).count();
        let revealed = commitments.values().filter(|c| c.revealed).count();

        CommitRevealStats {
            total_commitments: total,
            pending_reveals: pending,
            completed_reveals: revealed,
            enabled: self.enabled,
        }
    }
}

/// Statistics for commit-reveal system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRevealStats {
    pub total_commitments: usize,
    pub pending_reveals: usize,
    pub completed_reveals: usize,
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_commit_reveal_flow() {
        let manager = CommitRevealManager::new(true);

        let miner = "qnk1234567890123456789012345678901234567890123456789012345678901234";
        let nonce = 12345u64;
        let block_height = 100u64;
        let mut secret = [0u8; 32];
        secret[0] = 42;

        // Compute commitment
        let commitment_hash = CommitRevealManager::compute_commitment(
            nonce,
            miner,
            block_height,
            &secret,
        );

        // Register commitment
        let request = MiningCommitRequest {
            commitment_hash: hex::encode(commitment_hash),
            miner_address: miner.to_string(),
            block_height,
        };

        let commitment = manager.register_commitment(request).await.unwrap();
        assert!(!commitment.revealed);

        // Try early reveal (should fail)
        let reveal = MiningReveal {
            nonce,
            secret: hex::encode(secret),
            block_height,
            commitment_hash: hex::encode(commitment_hash),
        };

        let result = manager.process_reveal(reveal.clone(), miner, block_height + 1).await;
        assert!(result.is_err()); // Too early

        // Valid reveal after delay
        let result = manager.process_reveal(reveal, miner, block_height + MIN_REVEAL_DELAY).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), nonce);
    }

    #[test]
    fn test_temporal_mining_commit() {
        let miner = "qnk1234567890123456789012345678901234567890123456789012345678901234";
        let block_height = 100u64;
        let commitment_hash = [42u8; 32];
        let protected_reveal = vec![0u8; 256]; // Placeholder envelope
        let stark_proof = vec![0u8; 128];
        let key_commitment = [1u8; 32];

        let commit = TemporalMiningCommit::new(
            commitment_hash,
            protected_reveal,
            miner.to_string(),
            block_height,
            stark_proof,
            key_commitment,
        );

        // Check staggered reveal schedule
        assert_eq!(commit.reveal_schedule.len(), 3);
        assert_eq!(commit.reveal_schedule[0], (block_height + 2, 0));
        assert_eq!(commit.reveal_schedule[1], (block_height + 4, 1));
        assert_eq!(commit.reveal_schedule[2], (block_height + 6, 2));

        // Initially cannot reconstruct
        assert!(!commit.can_reconstruct);
        assert_eq!(commit.shares_released, 0);

        // Check release timing
        assert!(!commit.should_release_share(0, block_height + 1));
        assert!(commit.should_release_share(0, block_height + 2));
        assert!(commit.should_release_share(0, block_height + 3));
        assert!(!commit.should_release_share(1, block_height + 3));
        assert!(commit.should_release_share(1, block_height + 4));

        // Earliest reconstruction height
        assert_eq!(commit.earliest_reconstruction_height(), block_height + 4);
    }

    #[test]
    fn test_temporal_commit_share_release() {
        let mut commit = TemporalMiningCommit::new(
            [42u8; 32],
            vec![0u8; 256],
            "qnk1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            100,
            vec![0u8; 128],
            [1u8; 32],
        );

        // Release first share
        commit.record_share_release();
        assert_eq!(commit.shares_released, 1);
        assert!(!commit.can_reconstruct); // Need 2

        // Release second share
        commit.record_share_release();
        assert_eq!(commit.shares_released, 2);
        assert!(commit.can_reconstruct); // Now we can reconstruct
    }

    #[test]
    fn test_temporal_reveal_data() {
        let nonce = 12345u64;
        let secret = [42u8; 32];

        let data = TemporalRevealData::new(nonce, secret);
        let bytes = data.to_bytes();

        assert_eq!(bytes.len(), 40);

        let restored = TemporalRevealData::from_bytes(&bytes).unwrap();
        assert_eq!(restored.nonce, nonce);
        assert_eq!(restored.secret, secret);
    }

    #[test]
    fn test_temporal_commit_serialization() {
        let commit = TemporalMiningCommit::new(
            [42u8; 32],
            vec![0u8; 256],
            "qnk1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            100,
            vec![0u8; 128],
            [1u8; 32],
        );

        let bytes = commit.to_bytes().unwrap();
        let restored = TemporalMiningCommit::from_bytes(&bytes).unwrap();

        assert_eq!(commit.commitment_hash, restored.commitment_hash);
        assert_eq!(commit.miner_address, restored.miner_address);
        assert_eq!(commit.block_height, restored.block_height);
        assert_eq!(commit.reveal_schedule, restored.reveal_schedule);
    }

    #[test]
    fn test_next_scheduled_release() {
        let commit = TemporalMiningCommit::new(
            [42u8; 32],
            vec![0u8; 256],
            "qnk1234567890123456789012345678901234567890123456789012345678901234".to_string(),
            100,
            vec![0u8; 128],
            [1u8; 32],
        );

        // At height 100, next release is trustee 0 at 102
        let next = commit.next_scheduled_release(100);
        assert_eq!(next, Some((102, 0)));

        // At height 102, next release is trustee 1 at 104
        let next = commit.next_scheduled_release(102);
        assert_eq!(next, Some((104, 1)));

        // At height 105, next release is trustee 2 at 106
        let next = commit.next_scheduled_release(105);
        assert_eq!(next, Some((106, 2)));

        // At height 107, no more releases scheduled
        let next = commit.next_scheduled_release(107);
        assert_eq!(next, None);
    }
}
