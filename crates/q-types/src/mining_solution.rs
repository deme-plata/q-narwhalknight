//! P2P Mining Solution Types
//!
//! v2.3.5-beta: Gossipsub-based decentralized mining solution broadcasting.
//!
//! Solutions are broadcast to `/qnk/{network}/mining-solutions` for network-wide
//! reward credit with sub-50ms finality. The share ledger runs asynchronously
//! for audit/proof purposes only.

use serde::{Deserialize, Serialize};

/// Finality depth for network-consensus challenges
/// Canonical block = tip - FINALITY_DEPTH
/// v8.6.0: Reduced from 10 to 6 — DAG-Knight parallel finality makes 6 sufficient,
/// and faster challenge updates improve miner responsiveness
pub const MINING_FINALITY_DEPTH: u64 = 6;

/// P2P Mining Solution submission
/// Broadcast via gossipsub for network-wide reward credit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P2PMiningSubmission {
    /// Protocol version (for future upgrades)
    pub version: u8,

    /// Miner's wallet address (32-byte hash of public key)
    pub miner_address: [u8; 32],

    /// Mining solution hash (BLAKE3 hash of solution data)
    pub solution_hash: [u8; 32],

    /// Difficulty target met by this solution
    pub difficulty_target: [u8; 32],

    /// Block height this solution is for
    pub block_height: u64,

    /// Challenge hash this solution solves (from NetworkChallenge)
    pub challenge_hash: [u8; 32],

    /// Mining nonce that produced valid solution
    pub nonce: u64,

    /// VDF iterations completed (anti-grinding)
    pub vdf_iterations: u32,

    /// Timestamp when solution was found (milliseconds)
    pub timestamp_ms: u64,

    /// PeerId of the node that first received this solution
    pub origin_node_id: String,

    /// Ed25519/Dilithium5 signature over all fields above (64 bytes for Ed25519)
    pub signature: Vec<u8>,

    /// Signer's public key for verification
    pub signer_public_key: Vec<u8>,
}

impl P2PMiningSubmission {
    /// Create new submission (signature and signer_public_key must be filled by caller)
    pub fn new(
        miner_address: [u8; 32],
        solution_hash: [u8; 32],
        difficulty_target: [u8; 32],
        block_height: u64,
        challenge_hash: [u8; 32],
        nonce: u64,
        vdf_iterations: u32,
        origin_node_id: String,
    ) -> Self {
        Self {
            version: 1,
            miner_address,
            solution_hash,
            difficulty_target,
            block_height,
            challenge_hash,
            nonce,
            vdf_iterations,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            origin_node_id,
            signature: vec![0u8; 64], // Ed25519 signature size
            signer_public_key: Vec::new(),
        }
    }

    /// Get the data to be signed (all fields except signature and public key)
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(256);
        data.push(self.version);
        data.extend_from_slice(&self.miner_address);
        data.extend_from_slice(&self.solution_hash);
        data.extend_from_slice(&self.difficulty_target);
        data.extend_from_slice(&self.block_height.to_le_bytes());
        data.extend_from_slice(&self.challenge_hash);
        data.extend_from_slice(&self.nonce.to_le_bytes());
        data.extend_from_slice(&self.vdf_iterations.to_le_bytes());
        data.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        data.extend_from_slice(self.origin_node_id.as_bytes());
        data
    }

    /// Check if solution is within acceptable age (max 30 seconds)
    pub fn is_valid_age(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now.saturating_sub(self.timestamp_ms) <= max_age_ms
    }

    /// Get unique ID for deduplication (solution_hash is unique per solution)
    pub fn dedup_id(&self) -> [u8; 32] {
        self.solution_hash
    }
}

/// Network-consensus mining challenge
/// Derived from finalized block (tip - FINALITY_DEPTH)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkChallenge {
    /// Protocol version
    pub version: u32,

    /// Hash of the canonical (finalized) block
    pub canonical_block_hash: [u8; 32],

    /// Height of the canonical block
    pub canonical_height: u64,

    /// Current network difficulty target
    pub difficulty_target: [u8; 32],

    /// Required VDF iterations for anti-grinding
    pub vdf_iterations: u32,

    /// BLAKE3 hash of all above fields (used as challenge)
    pub challenge_hash: [u8; 32],

    /// Challenge expiration timestamp (milliseconds)
    pub expires_at_ms: u64,

    /// Node that generated this challenge
    pub generator_node_id: String,

    /// Signature from generator node (64 bytes for Ed25519)
    pub signature: Vec<u8>,
}

impl NetworkChallenge {
    /// Default challenge lifetime: 90 seconds
    /// v8.6.0: Increased from 60s to 90s — gives miners more time for valid submissions,
    /// especially on high-latency connections and Tor circuits
    pub const DEFAULT_LIFETIME_MS: u64 = 90_000;

    /// Create a new network challenge from a canonical block
    pub fn new(
        canonical_block_hash: [u8; 32],
        canonical_height: u64,
        difficulty_target: [u8; 32],
        vdf_iterations: u32,
        generator_node_id: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut challenge = Self {
            version: 1,
            canonical_block_hash,
            canonical_height,
            difficulty_target,
            vdf_iterations,
            challenge_hash: [0u8; 32],
            expires_at_ms: now + Self::DEFAULT_LIFETIME_MS,
            generator_node_id,
            signature: vec![0u8; 64], // Ed25519 signature size
        };

        // Compute challenge hash
        challenge.challenge_hash = challenge.compute_challenge_hash();
        challenge
    }

    /// Compute the challenge hash from fields
    pub fn compute_challenge_hash(&self) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.canonical_block_hash);
        hasher.update(&self.canonical_height.to_le_bytes());
        hasher.update(&self.difficulty_target);
        hasher.update(&self.vdf_iterations.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Check if challenge is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now < self.expires_at_ms && self.challenge_hash == self.compute_challenge_hash()
    }

    /// Check if this challenge matches another (same canonical block)
    pub fn matches(&self, other: &Self) -> bool {
        self.canonical_block_hash == other.canonical_block_hash
            && self.canonical_height == other.canonical_height
    }
}

/// Gossipsub message wrapper for mining P2P messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum P2PMiningMessage {
    /// A miner submitting a valid solution
    Solution(P2PMiningSubmission),

    /// A node announcing a new challenge
    Challenge(NetworkChallenge),

    /// Request for current challenge (from new/syncing nodes)
    ChallengeRequest {
        /// Requester's node ID
        requester_node_id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mining_submission_creation() {
        let submission = P2PMiningSubmission::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            1000,
            [4u8; 32],
            12345,
            100,
            "test-node".to_string(),
        );

        assert_eq!(submission.version, 1);
        assert_eq!(submission.block_height, 1000);
        assert_eq!(submission.nonce, 12345);
        assert!(submission.is_valid_age(60_000)); // Within 60 seconds
    }

    #[test]
    fn test_network_challenge() {
        let challenge = NetworkChallenge::new(
            [1u8; 32],
            999, // canonical_height = tip - 10
            [0u8; 32],
            100,
            "node-1".to_string(),
        );

        assert_eq!(challenge.version, 1);
        assert_eq!(challenge.canonical_height, 999);
        assert!(challenge.is_valid());
        assert_eq!(challenge.challenge_hash, challenge.compute_challenge_hash());
    }

    #[test]
    fn test_signing_data() {
        let submission = P2PMiningSubmission::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            1000,
            [4u8; 32],
            12345,
            100,
            "node".to_string(),
        );

        let data = submission.signing_data();
        assert!(data.len() > 100); // Should contain all field data
    }

    #[test]
    fn test_dedup_id() {
        let submission = P2PMiningSubmission::new(
            [1u8; 32],
            [99u8; 32], // unique solution hash
            [3u8; 32],
            1000,
            [4u8; 32],
            12345,
            100,
            "node".to_string(),
        );

        assert_eq!(submission.dedup_id(), [99u8; 32]);
    }
}
