// v0.9.14-beta: AEGIS-QL Signed P2P Block Synchronization
// Post-quantum secure block sync with cryptographic affirmation

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use q_types::QBlock;

/// v0.9.14-beta: AEGIS-QL signed block pack for P2P transmission (UNCOMPRESSED - legacy)
/// NOTE: This struct is kept for backwards compatibility but is inefficient (2 MB vs 600 KB)
/// Use SignedBlockPackCompressed for new implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedBlockPack {
    /// Blocks being transmitted (batch of up to 1000)
    pub blocks: Vec<QBlock>,

    /// Merkle root of all block hashes (for integrity)
    pub merkle_root: [u8; 32],

    /// AEGIS-QL post-quantum signature
    pub aegis_signature: q_aegis_ql::Signature,

    /// Sender's AEGIS-QL public key
    pub peer_public_key: q_aegis_ql::PublicKey,

    /// Timestamp (prevents replay attacks)
    pub timestamp: i64,

    /// Peer ID (for P2P routing, not cryptographic)
    pub peer_id: String,
}

/// v0.9.21-beta: COMPRESSED signed block pack for efficient P2P transmission
/// This reduces bandwidth from 2 MB (uncompressed) to ~603 KB (compressed + signature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedBlockPackCompressed {
    /// Compressed blocks using zstd level 3 (target: ~600 KB for 5000 blocks)
    pub compressed_blocks: Vec<u8>,

    /// Original block count (for validation)
    pub block_count: usize,

    /// Start and end heights (for range validation)
    pub start_height: u64,
    pub end_height: u64,

    /// Merkle root of UNCOMPRESSED block hashes (computed before compression)
    /// This ensures data integrity even after compression
    pub merkle_root: [u8; 32],

    /// AEGIS-QL Dilithium5 signature over (compressed_blocks || merkle_root || timestamp)
    /// Signing the compressed data ensures both compression integrity and block authenticity
    pub aegis_signature: q_aegis_ql::Signature,

    /// Sender's AEGIS-QL public key (for signature verification)
    pub peer_public_key: q_aegis_ql::PublicKey,

    /// Timestamp (prevents replay attacks, 5-minute window)
    pub timestamp: i64,

    /// Peer ID (for P2P routing and trust scoring)
    pub peer_id: String,

    /// Compression ratio (for metrics: original_size / compressed_size)
    pub compression_ratio: f64,
}

/// Certificate proving successful sync completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncAffirmationCertificate {
    /// Start height of synced range
    pub start_height: u64,

    /// End height of synced range
    pub end_height: u64,

    /// Hash of every block in range (for verification)
    pub block_hashes: Vec<[u8; 32]>,

    /// Merkle root of all block_hashes
    pub merkle_root: [u8; 32],

    /// AEGIS-QL signature over (start, end, merkle_root)
    pub aegis_signature: q_aegis_ql::Signature,

    /// Timestamp
    pub timestamp: i64,

    /// Syncing peer's public key
    pub syncer_public_key: q_aegis_ql::PublicKey,
}

/// Track peer reliability based on signature verification
#[derive(Debug, Clone)]
pub struct PeerTrustRegistry {
    /// Map peer_id -> trust metrics
    peers: DashMap<String, PeerTrustMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerTrustMetrics {
    /// Number of valid signed packs received
    pub valid_packs: u64,

    /// Number of invalid signatures detected
    pub invalid_signatures: u64,

    /// Number of blocks with wrong merkle roots
    pub merkle_failures: u64,

    /// v1.5.2-beta: Number of data failures (decompression, deserialization, etc.)
    pub data_failures: u64,

    /// Trust score (0.0 - 1.0)
    pub trust_score: f64,

    /// Last seen timestamp (any interaction)
    pub last_seen: i64,

    /// v10.2.10: Timestamp of most recent failure (for time-decay gating).
    /// Distinct from last_seen which updates on any interaction.
    /// Defaults to 0 for existing peers (immediately eligible for decay).
    #[serde(default)]
    pub last_failure_at: i64,

    /// v10.2.10: Consecutive successful chunk downloads (resets on any failure).
    /// Used for success-based trust recovery.
    #[serde(default)]
    pub consecutive_successes: u64,

    /// Peer's AEGIS-QL public key
    pub public_key: q_aegis_ql::PublicKey,
}

impl PeerTrustRegistry {
    pub fn new() -> Self {
        Self {
            peers: DashMap::new(),
        }
    }

    /// Record a valid signed pack from peer
    pub fn record_valid_pack(&self, peer_id: &str, public_key: q_aegis_ql::PublicKey) {
        let mut entry = self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            PeerTrustMetrics {
                valid_packs: 0,
                invalid_signatures: 0,
                merkle_failures: 0,
                data_failures: 0,
                trust_score: 0.5, // Start neutral
                last_seen: chrono::Utc::now().timestamp(),
                last_failure_at: 0,
                consecutive_successes: 0,
                public_key,
            }
        });

        entry.valid_packs += 1;
        entry.last_seen = chrono::Utc::now().timestamp();

        // Update trust score (exponential moving average)
        // valid_packs / (valid_packs + invalid_signatures + merkle_failures)
        let total_interactions = entry.valid_packs + entry.invalid_signatures + entry.merkle_failures;
        entry.trust_score = entry.valid_packs as f64 / total_interactions as f64;

        tracing::info!(
            "✅ [AEGIS-QL] Peer {} trust: {:.2}% ({} valid packs)",
            &peer_id[..8],
            entry.trust_score * 100.0,
            entry.valid_packs
        );
    }

    /// Record an invalid signature from peer (CRITICAL)
    pub fn record_invalid_signature(&self, peer_id: &str, public_key: q_aegis_ql::PublicKey) {
        let mut entry = self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            PeerTrustMetrics {
                valid_packs: 0,
                invalid_signatures: 0,
                merkle_failures: 0,
                data_failures: 0,
                trust_score: 0.5,
                last_seen: chrono::Utc::now().timestamp(),
                last_failure_at: 0,
                consecutive_successes: 0,
                public_key,
            }
        });

        entry.invalid_signatures += 1;
        entry.last_seen = chrono::Utc::now().timestamp();
        entry.last_failure_at = chrono::Utc::now().timestamp();
        entry.consecutive_successes = 0;

        // Severely penalize trust score for invalid signatures
        let total_interactions = entry.valid_packs + entry.invalid_signatures + entry.merkle_failures;
        entry.trust_score = entry.valid_packs as f64 / total_interactions as f64;

        tracing::error!(
            "🚨 [AEGIS-QL] INVALID SIGNATURE from peer {}! Trust: {:.2}% ({} failures)",
            &peer_id[..8],
            entry.trust_score * 100.0,
            entry.invalid_signatures
        );
    }

    /// Record a merkle root mismatch from peer
    pub fn record_merkle_failure(&self, peer_id: &str, public_key: q_aegis_ql::PublicKey) {
        let mut entry = self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            PeerTrustMetrics {
                valid_packs: 0,
                invalid_signatures: 0,
                merkle_failures: 0,
                data_failures: 0,
                trust_score: 0.5,
                last_seen: chrono::Utc::now().timestamp(),
                last_failure_at: 0,
                consecutive_successes: 0,
                public_key,
            }
        });

        entry.merkle_failures += 1;
        entry.last_seen = chrono::Utc::now().timestamp();
        entry.last_failure_at = chrono::Utc::now().timestamp();
        entry.consecutive_successes = 0;

        let total_failures = entry.invalid_signatures + entry.merkle_failures + entry.data_failures;
        let total_interactions = entry.valid_packs + total_failures;
        entry.trust_score = entry.valid_packs as f64 / total_interactions.max(1) as f64;

        tracing::warn!(
            "⚠️ [AEGIS-QL] Merkle failure from peer {}. Trust: {:.2}% ({} merkle failures)",
            &peer_id[..8.min(peer_id.len())],
            entry.trust_score * 100.0,
            entry.merkle_failures
        );
    }

    /// v1.5.2-beta: Record a data failure from peer (decompression, deserialization, etc.)
    /// This reduces trust score significantly to prevent repeated requests to bad peers
    pub fn record_data_failure(&self, peer_id: &str) {
        let mut entry = self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            PeerTrustMetrics {
                valid_packs: 0,
                invalid_signatures: 0,
                merkle_failures: 0,
                data_failures: 0,
                trust_score: 0.5,
                last_seen: chrono::Utc::now().timestamp(),
                last_failure_at: 0,
                consecutive_successes: 0,
                // Default empty key for unknown peers
                public_key: q_aegis_ql::PublicKey { a: Vec::new(), t: Vec::new() },
            }
        });

        entry.data_failures += 1;
        entry.last_seen = chrono::Utc::now().timestamp();
        entry.last_failure_at = chrono::Utc::now().timestamp();
        entry.consecutive_successes = 0; // v10.2.10: Reset on failure

        // v1.5.2-beta: Heavily penalize data failures (likely version mismatch)
        let total_failures = entry.invalid_signatures + entry.merkle_failures + entry.data_failures;
        let total_interactions = entry.valid_packs + total_failures;
        // Data failures are weighted 2x to quickly identify incompatible nodes
        let weighted_valid = entry.valid_packs as f64;
        let weighted_failures = (entry.invalid_signatures + entry.merkle_failures + entry.data_failures * 2) as f64;
        entry.trust_score = weighted_valid / (weighted_valid + weighted_failures).max(1.0);

        tracing::warn!(
            "🔧 [AEGIS-QL] Data failure from peer {}. Trust: {:.2}% ({} data failures, {} total)",
            &peer_id[..8.min(peer_id.len())],
            entry.trust_score * 100.0,
            entry.data_failures,
            total_interactions
        );
    }

    /// Check if peer should be banned (trust score < 20%)
    pub fn should_ban_peer(&self, peer_id: &str) -> bool {
        if let Some(metrics) = self.peers.get(peer_id) {
            metrics.trust_score < 0.2
        } else {
            false
        }
    }

    /// Get peer trust score (returns None if peer not known)
    pub fn get_trust_score(&self, peer_id: &str) -> Option<f64> {
        self.peers.get(peer_id).map(|m| m.trust_score)
    }

    /// Get all trusted peers (trust score >= 80%)
    pub fn get_trusted_peers(&self) -> Vec<String> {
        self.peers
            .iter()
            .filter(|entry| entry.value().trust_score >= 0.8)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// v10.2.10: Record a successful chunk download from a peer.
    /// After 5 consecutive successes, reduce failure counters by 1.
    /// The consecutive_successes counter resets on any failure, so this only
    /// triggers for peers that have genuinely recovered.
    pub fn record_successful_chunk(&self, peer_id: &str) {
        if let Some(mut entry) = self.peers.get_mut(peer_id) {
            entry.valid_packs += 1;
            entry.consecutive_successes += 1;
            entry.last_seen = chrono::Utc::now().timestamp();

            if entry.consecutive_successes >= 5 && entry.trust_score < 0.5 {
                entry.invalid_signatures = entry.invalid_signatures.saturating_sub(1);
                entry.merkle_failures = entry.merkle_failures.saturating_sub(1);
                entry.data_failures = entry.data_failures.saturating_sub(1);
                entry.consecutive_successes = 0; // Reset — next recovery needs 5 more

                Self::recalculate_trust(&mut *entry);

                tracing::info!(
                    "🔄 [AEGIS-QL] Peer {} trust recovered to {:.2}% after 5 consecutive successes",
                    &peer_id[..8.min(peer_id.len())],
                    entry.trust_score * 100.0
                );
            } else {
                Self::recalculate_trust(&mut *entry);
            }
        }
    }

    /// v10.2.10: Time-decay failure counters.
    /// Halves failure counters for peers whose last failure was >5 minutes ago.
    /// Uses last_failure_at (not last_seen) so actively-used peers with no
    /// recent failures also benefit from decay.
    pub fn apply_time_decay(&self) {
        let now = chrono::Utc::now().timestamp();
        let decay_threshold_secs = 300; // 5 minutes
        let mut decayed_count = 0;

        for mut entry in self.peers.iter_mut() {
            let since_last_failure = now - entry.last_failure_at;
            if since_last_failure > decay_threshold_secs && entry.trust_score < 0.5 {
                entry.invalid_signatures /= 2;
                entry.merkle_failures /= 2;
                entry.data_failures /= 2;
                Self::recalculate_trust(&mut *entry);
                decayed_count += 1;
            }
        }

        if decayed_count > 0 {
            tracing::info!(
                "🔄 [AEGIS-QL] Time-decay applied to {} peer(s) with stale failures",
                decayed_count
            );
        }
    }

    /// v10.2.10: Remove peers not seen in over 1 hour.
    pub fn cleanup_stale_peers(&self) {
        let now = chrono::Utc::now().timestamp();
        let before = self.peers.len();
        self.peers.retain(|_, entry| (now - entry.last_seen) < 3600);
        let removed = before - self.peers.len();
        if removed > 0 {
            tracing::info!(
                "🧹 [AEGIS-QL] Cleaned up {} stale peer(s) (not seen in >1 hour)",
                removed
            );
        }
    }

    /// v10.2.10: Recalculate trust score from current counters.
    fn recalculate_trust(metrics: &mut PeerTrustMetrics) {
        let weighted_valid = metrics.valid_packs as f64;
        let weighted_failures = (metrics.invalid_signatures
            + metrics.merkle_failures
            + metrics.data_failures * 2) as f64;
        metrics.trust_score = if weighted_valid + weighted_failures > 0.0 {
            weighted_valid / (weighted_valid + weighted_failures)
        } else {
            0.5 // Reset to neutral if all counters decayed to 0
        };
    }
}

impl Default for PeerTrustRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute merkle root of block hashes
pub fn compute_merkle_root(hashes: &[[u8; 32]]) -> [u8; 32] {
    use sha2::{Sha256, Digest};

    if hashes.is_empty() {
        return [0u8; 32];
    }

    if hashes.len() == 1 {
        return hashes[0];
    }

    // Build merkle tree bottom-up
    let mut current_level: Vec<[u8; 32]> = hashes.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&chunk[0]);

            if chunk.len() == 2 {
                hasher.update(&chunk[1]);
            } else {
                // Duplicate last hash if odd number
                hasher.update(&chunk[0]);
            }

            let hash = hasher.finalize();
            let mut result = [0u8; 32];
            result.copy_from_slice(&hash);
            next_level.push(result);
        }

        current_level = next_level;
    }

    current_level[0]
}

/// Verify timestamp is within acceptable window (5 minutes)
pub fn verify_timestamp(timestamp: i64) -> bool {
    let now = chrono::Utc::now().timestamp();
    let diff = (now - timestamp).abs();

    // Allow 5-minute window for clock skew
    diff < 300
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_root_single() {
        let hash = [42u8; 32];
        let root = compute_merkle_root(&[hash]);
        assert_eq!(root, hash);
    }

    #[test]
    fn test_merkle_root_two() {
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];
        let root = compute_merkle_root(&[hash1, hash2]);

        // Should be SHA256(hash1 || hash2)
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&hash1);
        hasher.update(&hash2);
        let expected = hasher.finalize();

        assert_eq!(&root[..], &expected[..]);
    }

    #[test]
    fn test_peer_trust_registry() {
        let registry = PeerTrustRegistry::new();
        let public_key = q_aegis_ql::PublicKey::default();

        // Start with neutral trust
        registry.record_valid_pack("peer1", public_key.clone());
        assert!(registry.get_trust_score("peer1").unwrap() >= 0.9);

        // Record invalid signature
        registry.record_invalid_signature("peer1", public_key.clone());
        let trust = registry.get_trust_score("peer1").unwrap();
        assert!(trust < 0.6); // Trust should drop significantly

        // Should ban peer if trust drops too low
        registry.record_invalid_signature("peer1", public_key.clone());
        registry.record_invalid_signature("peer1", public_key.clone());
        assert!(registry.should_ban_peer("peer1"));
    }

    #[test]
    fn test_timestamp_verification() {
        let now = chrono::Utc::now().timestamp();

        // Current time should be valid
        assert!(verify_timestamp(now));

        // 2 minutes ago should be valid
        assert!(verify_timestamp(now - 120));

        // 6 minutes ago should be invalid
        assert!(!verify_timestamp(now - 360));

        // 2 minutes in future should be valid (clock skew)
        assert!(verify_timestamp(now + 120));

        // 6 minutes in future should be invalid
        assert!(!verify_timestamp(now + 360));
    }
}
