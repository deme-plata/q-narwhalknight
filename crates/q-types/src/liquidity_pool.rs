/// Liquidity Pool P2P Broadcasting Module
/// v0.6.0-beta: DEX Decentralization Phase 2
///
/// This module provides data structures and cryptographic functions for broadcasting
/// liquidity pools across the P2P network using gossipsub.
///
/// Key features:
/// - Deterministic pool ID generation (prevents concurrent creation conflicts)
/// - Ed25519 signature verification for pool announcements
/// - Domain-tagged signing for security (prevents signature replay attacks)
/// - Active sync protocol for new nodes (request-response)
/// - Rate limiting with automatic cleanup

use crate::{Amount, NodeId};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Domain tag for pool announcement signatures (prevents cross-context replay attacks)
const POOL_ANNOUNCEMENT_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-POOL-ANNOUNCEMENT-V1";

/// Domain tag for pool sync request signatures
const POOL_SYNC_REQUEST_DOMAIN: &[u8] = b"Q-NARWHALKNIGHT-POOL-SYNC-REQUEST-V1";

/// Liquidity pool announcement message for P2P broadcasting
/// This is the primary data structure broadcast on the `/qnk/liquidity-pools` gossipsub topic
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolAnnouncement {
    /// Deterministic pool ID: Hash(token0_addr || token1_addr) where token0 < token1
    pub pool_id: [u8; 32],

    /// First token address (lexicographically smaller)
    pub token0: [u8; 32],

    /// Second token address (lexicographically larger)
    pub token1: [u8; 32],

    /// Reserve of token0 (in base units, 8 decimals for QUG/QUGUSD)
    pub reserve0: Amount,

    /// Reserve of token1 (in base units)
    pub reserve1: Amount,

    /// Total LP token supply (calculated using Uniswap V2 formula)
    pub lp_token_supply: Amount,

    /// Pool creator's wallet address (32-byte Ed25519 public key)
    pub creator: NodeId,

    /// Unix timestamp when pool was created/updated
    pub timestamp: u64,

    /// Ed25519 signature of the announcement (64 bytes)
    /// Signs: domain_tag || pool_id || token0 || token1 || reserve0 || reserve1 || lp_token_supply || timestamp
    pub signature: Vec<u8>,

    /// Announcement version (for future protocol upgrades)
    pub version: u8,
}

impl PoolAnnouncement {
    /// Create a new pool announcement (unsigned)
    pub fn new(
        token0: [u8; 32],
        token1: [u8; 32],
        reserve0: Amount,
        reserve1: Amount,
        lp_token_supply: Amount,
        creator: NodeId,
        timestamp: u64,
    ) -> Self {
        // Ensure canonical ordering (token0 < token1)
        let (canonical_token0, canonical_token1, canonical_reserve0, canonical_reserve1) =
            if token0 < token1 {
                (token0, token1, reserve0, reserve1)
            } else {
                (token1, token0, reserve1, reserve0)
            };

        let pool_id = Self::generate_pool_id(&canonical_token0, &canonical_token1);

        Self {
            pool_id,
            token0: canonical_token0,
            token1: canonical_token1,
            reserve0: canonical_reserve0,
            reserve1: canonical_reserve1,
            lp_token_supply,
            creator,
            timestamp,
            signature: Vec::new(),
            version: 1,
        }
    }

    /// Generate deterministic pool ID from token addresses
    /// Formula: SHA3-256(token0 || token1) where token0 < token1
    /// This prevents concurrent pool creation conflicts across nodes
    pub fn generate_pool_id(token0: &[u8; 32], token1: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"Q-NARWHALKNIGHT-POOL-ID-V1");
        hasher.update(token0);
        hasher.update(token1);
        hasher.finalize().into()
    }

    /// Get the canonical signing message for this announcement
    /// Format: domain_tag || pool_id || token0 || token1 || reserve0_le || reserve1_le || lp_supply_le || timestamp_le
    fn signing_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(POOL_ANNOUNCEMENT_DOMAIN);
        message.extend_from_slice(&self.pool_id);
        message.extend_from_slice(&self.token0);
        message.extend_from_slice(&self.token1);
        message.extend_from_slice(&self.reserve0.to_le_bytes());
        message.extend_from_slice(&self.reserve1.to_le_bytes());
        message.extend_from_slice(&self.lp_token_supply.to_le_bytes());
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        message
    }

    /// Sign this announcement with the creator's Ed25519 private key
    /// Returns an error if signing fails
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let signature = crate::signature_verification::sign_ed25519(&message, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify the Ed25519 signature on this announcement
    /// Returns Ok(()) if signature is valid, Err otherwise
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!(
                "Invalid signature length: expected 64 bytes, got {}",
                self.signature.len()
            ));
        }

        let message = self.signing_message();

        // Use the existing Ed25519 verification from signature_verification module
        crate::signature_verification::verify_block_signature(
            &self.signature,
            // We need to hash the message to get a 32-byte block hash
            &Sha3_256::digest(&message).into(),
            &self.creator,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
        .map_err(|e| anyhow!("Pool announcement signature verification failed: {}", e))
    }

    /// Verify pool announcement is well-formed
    /// Checks:
    /// - Pool ID matches hash of (token0, token1)
    /// - Token ordering is canonical (token0 < token1)
    /// - Reserves are non-zero
    /// - LP token supply is non-zero
    /// - Timestamp is reasonable (not too far in future)
    pub fn verify_structure(&self) -> Result<()> {
        // Verify pool ID is correct
        let expected_pool_id = Self::generate_pool_id(&self.token0, &self.token1);
        if self.pool_id != expected_pool_id {
            return Err(anyhow!(
                "Pool ID mismatch: expected {:?}, got {:?}",
                expected_pool_id,
                self.pool_id
            ));
        }

        // Verify canonical token ordering
        if self.token0 >= self.token1 {
            return Err(anyhow!(
                "Invalid token ordering: token0 must be < token1 (got token0={:?}, token1={:?})",
                self.token0,
                self.token1
            ));
        }

        // Verify reserves are non-zero
        if self.reserve0 == 0 {
            return Err(anyhow!("Reserve0 must be non-zero"));
        }
        if self.reserve1 == 0 {
            return Err(anyhow!("Reserve1 must be non-zero"));
        }

        // Verify LP token supply is non-zero
        if self.lp_token_supply == 0 {
            return Err(anyhow!("LP token supply must be non-zero"));
        }

        // Verify timestamp is not too far in the future (allow 5 minute clock skew)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if self.timestamp > now + 300 {
            return Err(anyhow!(
                "Timestamp too far in future: {} vs now {}",
                self.timestamp,
                now
            ));
        }

        Ok(())
    }

    /// Full verification: structure + signature
    pub fn verify(&self) -> Result<()> {
        self.verify_structure()?;
        self.verify_signature()?;
        Ok(())
    }
}

/// Request to synchronize liquidity pools from a peer
/// Used in the active sync protocol (libp2p request-response)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolSyncRequest {
    /// Requesting node's public key
    pub requester: NodeId,

    /// Unix timestamp of the request
    pub timestamp: u64,

    /// Optional: Only sync pools created/updated after this timestamp
    /// If None, sync all pools
    pub since_timestamp: Option<u64>,

    /// Ed25519 signature of the request
    /// Signs: domain_tag || requester || timestamp || since_timestamp
    pub signature: Vec<u8>,
}

impl PoolSyncRequest {
    /// Create a new pool sync request (unsigned)
    pub fn new(requester: NodeId, since_timestamp: Option<u64>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            requester,
            timestamp,
            since_timestamp,
            signature: Vec::new(),
        }
    }

    /// Get the signing message for this request
    fn signing_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(POOL_SYNC_REQUEST_DOMAIN);
        message.extend_from_slice(&self.requester);
        message.extend_from_slice(&self.timestamp.to_le_bytes());
        if let Some(since) = self.since_timestamp {
            message.extend_from_slice(&since.to_le_bytes());
        }
        message
    }

    /// Sign this request with the requester's Ed25519 private key
    #[cfg(feature = "signing")]
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let message = self.signing_message();
        let signature = crate::signature_verification::sign_ed25519(&message, signing_key);
        self.signature = signature;
        Ok(())
    }

    /// Verify the signature on this request
    pub fn verify_signature(&self) -> Result<()> {
        if self.signature.len() != 64 {
            return Err(anyhow!("Invalid signature length: expected 64 bytes"));
        }

        let message = self.signing_message();
        crate::signature_verification::verify_block_signature(
            &self.signature,
            &Sha3_256::digest(&message).into(),
            &self.requester,
            crate::block::SignaturePhase::Phase0Ed25519,
        )
        .map_err(|e| anyhow!("Pool sync request signature verification failed: {}", e))
    }
}

/// Response to a pool synchronization request
/// Contains all pools matching the request criteria
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolSyncResponse {
    /// All pool announcements matching the request
    pub pools: Vec<PoolAnnouncement>,

    /// Unix timestamp of the response
    pub timestamp: u64,

    /// Responding node's public key
    pub responder: NodeId,
}

impl PoolSyncResponse {
    /// Create a new pool sync response
    pub fn new(pools: Vec<PoolAnnouncement>, responder: NodeId) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            pools,
            timestamp,
            responder,
        }
    }

    /// Verify all pools in the response
    pub fn verify_all_pools(&self) -> Result<()> {
        for pool in &self.pools {
            pool.verify()?;
        }
        Ok(())
    }
}

/// Rate limiter for pool announcements (prevents spam)
/// Tracks announcements per peer with automatic cleanup
pub struct PoolAnnouncementRateLimiter {
    /// Map: peer_id -> (announcement_count, window_start_timestamp)
    limits: std::collections::HashMap<NodeId, (u32, u64)>,

    /// Maximum announcements per peer per window
    max_per_window: u32,

    /// Time window in seconds (e.g., 60 for 1 minute)
    window_seconds: u64,
}

impl PoolAnnouncementRateLimiter {
    /// Create a new rate limiter
    /// Default: 10 announcements per peer per minute
    pub fn new(max_per_window: u32, window_seconds: u64) -> Self {
        Self {
            limits: std::collections::HashMap::new(),
            max_per_window,
            window_seconds,
        }
    }

    /// Check if a peer is allowed to announce a pool
    /// Returns true if allowed, false if rate limit exceeded
    pub fn check_and_increment(&mut self, peer_id: &NodeId) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = self.limits.entry(*peer_id).or_insert((0, now));

        // Check if we're in a new window
        if now >= entry.1 + self.window_seconds {
            // New window - reset counter
            entry.0 = 1;
            entry.1 = now;
            return true;
        }

        // Same window - check limit
        if entry.0 >= self.max_per_window {
            return false; // Rate limit exceeded
        }

        // Increment counter
        entry.0 += 1;
        true
    }

    /// Cleanup old entries (call periodically to prevent memory leak)
    /// Removes entries older than 2x the window size
    pub fn cleanup(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cutoff = now.saturating_sub(self.window_seconds * 2);

        self.limits.retain(|_, (_, window_start)| *window_start >= cutoff);
    }

    /// Get number of tracked peers
    pub fn tracked_peers(&self) -> usize {
        self.limits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic per-test signing key. ed25519-dalek 2.x dropped
    /// `SigningKey::generate`; we use `from_bytes` with a tag-derived seed
    /// so each test gets a stable, distinct key without an RNG dependency.
    #[cfg(feature = "signing")]
    fn signing_key_from_tag(tag: u64) -> ed25519_dalek::SigningKey {
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&tag.to_le_bytes());
        ed25519_dalek::SigningKey::from_bytes(&seed)
    }

    #[test]
    fn test_pool_id_generation_is_deterministic() {
        let token0 = [1u8; 32];
        let token1 = [2u8; 32];

        let id1 = PoolAnnouncement::generate_pool_id(&token0, &token1);
        let id2 = PoolAnnouncement::generate_pool_id(&token0, &token1);

        assert_eq!(id1, id2, "Pool ID generation should be deterministic");
    }

    #[test]
    fn test_pool_id_generation_is_commutative() {
        let token0 = [1u8; 32];
        let token1 = [2u8; 32];

        let id1 = PoolAnnouncement::generate_pool_id(&token0, &token1);
        let id2 = PoolAnnouncement::generate_pool_id(&token1, &token0);

        assert_eq!(
            id1, id2,
            "Pool ID should be the same regardless of token order"
        );
    }

    #[test]
    fn test_pool_announcement_canonical_ordering() {
        let token0 = [2u8; 32]; // Larger
        let token1 = [1u8; 32]; // Smaller

        let announcement = PoolAnnouncement::new(
            token0,
            token1,
            1000,
            2000,
            1414, // sqrt(1000 * 2000) ≈ 1414
            [0u8; 32],
            1700000000,
        );

        // Should be reordered to token1, token0
        assert_eq!(announcement.token0, token1);
        assert_eq!(announcement.token1, token0);
        assert_eq!(announcement.reserve0, 2000); // Swapped
        assert_eq!(announcement.reserve1, 1000); // Swapped
    }

    #[test]
    fn test_pool_announcement_structure_verification() {
        let token0 = [1u8; 32];
        let token1 = [2u8; 32];

        let announcement = PoolAnnouncement::new(
            token0, token1, 1000, 2000, 1414, [0u8; 32], 1700000000,
        );

        // Should pass structure verification (signature check will fail, but that's okay)
        assert!(announcement.verify_structure().is_ok());
    }

    #[test]
    fn test_pool_announcement_invalid_zero_reserves() {
        let token0 = [1u8; 32];
        let token1 = [2u8; 32];

        let mut announcement =
            PoolAnnouncement::new(token0, token1, 0, 2000, 1000, [0u8; 32], 1700000000);

        // Should fail - reserve0 is zero
        assert!(announcement.verify_structure().is_err());

        announcement.reserve0 = 1000;
        announcement.reserve1 = 0;

        // Should fail - reserve1 is zero
        assert!(announcement.verify_structure().is_err());
    }

    #[test]
    fn test_rate_limiter_basic() {
        let mut limiter = PoolAnnouncementRateLimiter::new(3, 60);
        let peer_id = [1u8; 32];

        // First 3 announcements should succeed
        assert!(limiter.check_and_increment(&peer_id));
        assert!(limiter.check_and_increment(&peer_id));
        assert!(limiter.check_and_increment(&peer_id));

        // 4th announcement should fail
        assert!(!limiter.check_and_increment(&peer_id));
    }

    #[test]
    fn test_rate_limiter_cleanup() {
        let mut limiter = PoolAnnouncementRateLimiter::new(10, 1); // 1-second window
        let peer1 = [1u8; 32];
        let peer2 = [2u8; 32];

        limiter.check_and_increment(&peer1);
        limiter.check_and_increment(&peer2);

        assert_eq!(limiter.tracked_peers(), 2);

        // Wait for entries to become old (3+ seconds)
        std::thread::sleep(std::time::Duration::from_secs(3));

        limiter.cleanup();

        // Old entries should be removed
        assert_eq!(limiter.tracked_peers(), 0);
    }

    #[cfg(feature = "signing")]
    #[test]
    fn test_pool_announcement_signature() {
        let signing_key = signing_key_from_tag(0xA001);
        let public_key: [u8; 32] = signing_key.verifying_key().to_bytes();

        let token0 = [1u8; 32];
        let token1 = [2u8; 32];

        let mut announcement = PoolAnnouncement::new(
            token0,
            token1,
            1000000000, // 10 QUG (8 decimals)
            2000000000, // 20 QUG
            1414213562, // sqrt(1000000000 * 2000000000)
            public_key,
            1700000000,
        );

        // Sign the announcement
        announcement.sign(&signing_key).unwrap();

        // Verify the signature
        assert!(announcement.verify_signature().is_ok());

        // Verify full announcement
        assert!(announcement.verify().is_ok());
    }

    #[cfg(feature = "signing")]
    #[test]
    fn test_pool_sync_request_signature() {
        let signing_key = signing_key_from_tag(0xA002);
        let public_key: [u8; 32] = signing_key.verifying_key().to_bytes();

        let mut request = PoolSyncRequest::new(public_key, Some(1700000000));

        // Sign the request
        request.sign(&signing_key).unwrap();

        // Verify the signature
        assert!(request.verify_signature().is_ok());
    }
}
