/// v1.4.11-beta: Commit-Reveal Mining Protection
///
/// Provides cryptographic time-locks for mining submissions to prevent:
/// - Front-running attacks (seeing nonce before reveal)
/// - MEV (Miner Extractable Value) theft
/// - Block withholding with instant reveal
///
/// Flow:
/// 1. Miner commits: H(nonce || miner_address || block_height || secret)
/// 2. Wait REVEAL_DELAY blocks (2-5 blocks)
/// 3. Miner reveals: (nonce, secret) - verified against commitment
///
/// Security Properties:
/// - Binding: Cannot change nonce after commitment
/// - Hiding: Cannot determine nonce from commitment (secret provides entropy)
/// - Time-locked: Must wait before revealing (prevents front-running)

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
}
