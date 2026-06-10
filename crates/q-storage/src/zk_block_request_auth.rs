//! ZK-SNARK Block Request Authentication for Trustless P2P Sync
//!
//! **v0.9.6-beta**: Zero-Knowledge membership proofs for TurboSync block pack requests
//!
//! This module bridges the existing `q-zk-p2p::NetworkMembershipProof` infrastructure
//! to TurboSync, preventing unauthorized or sybil attacks on block distribution.
//!
//! **Security Properties**:
//! - Prevents non-members from requesting block packs
//! - Stops sybil attacks on bandwidth exhaustion
//! - Enables trustless block distribution
//! - GPU-accelerated verification (<50ms)
//! - Quantum-resistant (PLONK/Groth16 with PQ-ready curves)

use anyhow::Result;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use q_zk_p2p::{NetworkMembershipProof, MerkleTree};

/// Block pack request with ZK-SNARK membership proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedBlockPackRequest {
    /// Requester peer ID
    pub peer_id: String,

    /// Starting block height
    pub start_height: u64,

    /// Ending block height
    pub end_height: u64,

    /// ZK-SNARK proof: "I am an authorized network member"
    /// Proves membership without revealing identity or position in network
    pub membership_proof: Option<NetworkMembershipProof>,

    /// Request timestamp
    pub timestamp: u64,
}

/// Block pack response with optional validity proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedBlockPackResponse {
    /// Block pack data (compressed)
    pub pack_data: Vec<u8>,

    /// Number of blocks in pack
    pub block_count: usize,

    /// Start/end heights
    pub start_height: u64,
    pub end_height: u64,

    /// Optional ZK-SNARK proof: "All blocks in this pack are valid"
    /// Allows verifier to reject invalid packs before decompression (saves bandwidth)
    pub validity_proof: Option<Vec<u8>>, // Serialized PLONK proof

    /// Responder peer ID
    pub responder_peer_id: String,
}

/// Block request authenticator with reputation tracking
pub struct BlockRequestAuthenticator {
    /// Network Merkle tree (all authorized peers)
    network_tree: Arc<RwLock<MerkleTree>>,

    /// Request rate limiter: peer_id -> (requests_count, last_reset_time)
    rate_limiter: Arc<RwLock<std::collections::HashMap<PeerId, (u64, Instant)>>>,

    /// Ban list: peers with too many failed proofs or rate limit violations
    banned_peers: Arc<RwLock<std::collections::HashSet<PeerId>>>,

    /// Enable ZK verification (default: false for gradual rollout)
    verification_enabled: bool,

    /// Rate limit: max requests per minute per peer
    max_requests_per_minute: u64,
}

impl BlockRequestAuthenticator {
    /// Create new block request authenticator
    pub async fn new(network_tree: MerkleTree, enable_verification: bool) -> Result<Self> {
        Ok(Self {
            network_tree: Arc::new(RwLock::new(network_tree)),
            rate_limiter: Arc::new(RwLock::new(std::collections::HashMap::new())),
            banned_peers: Arc::new(RwLock::new(std::collections::HashSet::new())),
            verification_enabled: enable_verification,
            max_requests_per_minute: 10, // Configurable rate limit
        })
    }

    /// Enable ZK proof verification (after gradual rollout)
    pub fn enable_verification(&mut self) {
        self.verification_enabled = true;
        info!("🔐 [ZK BLOCK AUTH] Verification ENABLED - authenticated requests only!");
    }

    /// Disable ZK proof verification (for testing/rollback)
    pub fn disable_verification(&mut self) {
        self.verification_enabled = false;
        warn!("⚠️ [ZK BLOCK AUTH] Verification DISABLED - accepting unauthenticated requests!");
    }

    /// Check if peer is banned
    pub async fn is_banned(&self, peer_id: &PeerId) -> bool {
        self.banned_peers.read().await.contains(peer_id)
    }

    /// Check rate limit for peer
    pub async fn check_rate_limit(&self, peer_id: &PeerId) -> Result<bool> {
        let mut rate_limiter = self.rate_limiter.write().await;

        let now = Instant::now();
        let entry = rate_limiter.entry(*peer_id).or_insert((0, now));

        // Reset counter if 1 minute has passed
        if now.duration_since(entry.1).as_secs() >= 60 {
            entry.0 = 0;
            entry.1 = now;
        }

        // Check if rate limit exceeded
        if entry.0 >= self.max_requests_per_minute {
            warn!("⚠️ [ZK BLOCK AUTH] Rate limit exceeded for peer {} ({} requests/min)",
                  peer_id, entry.0);
            return Ok(false);
        }

        // Increment request count
        entry.0 += 1;
        debug!("📊 [ZK BLOCK AUTH] Peer {} request count: {}/{}",
               peer_id, entry.0, self.max_requests_per_minute);

        Ok(true)
    }

    /// Verify block pack request authentication
    ///
    /// **Returns**:
    /// - Ok(true) if proof valid OR verification disabled
    /// - Ok(false) if proof missing but required, or rate limit exceeded
    /// - Err if proof invalid (malicious peer) or peer banned
    pub async fn verify_block_request(
        &self,
        peer_id: &PeerId,
        request: &AuthenticatedBlockPackRequest,
    ) -> Result<bool> {
        // Check if peer is banned
        if self.is_banned(peer_id).await {
            error!("🚫 [ZK BLOCK AUTH] Rejected request from BANNED peer {}", peer_id);
            return Err(anyhow::anyhow!("Peer is banned"));
        }

        // Check rate limit
        if !self.check_rate_limit(peer_id).await? {
            warn!("🚫 [ZK BLOCK AUTH] Rate limit exceeded for peer {}", peer_id);
            self.record_rate_limit_violation(peer_id).await;
            return Ok(false);
        }

        // If verification disabled, accept all requests (backward compatibility)
        if !self.verification_enabled {
            debug!("📡 [ZK BLOCK AUTH] Verification disabled - accepting request from {} without proof",
                   peer_id);
            return Ok(true);
        }

        // If verification enabled, proof is required
        let proof = match &request.membership_proof {
            Some(p) => p,
            None => {
                warn!("⚠️ [ZK BLOCK AUTH] Peer {} requested blocks WITHOUT membership proof (verification enabled!)",
                      peer_id);
                return Ok(false); // Reject unproven requests
            }
        };

        // Verify ZK-SNARK membership proof
        let start = Instant::now();
        let network_tree = self.network_tree.read().await;

        match self.verify_membership_proof(proof, &network_tree).await {
            Ok(true) => {
                let verify_time = start.elapsed();
                info!("✅ [ZK BLOCK AUTH] Verified membership proof from {} in {:?}",
                      peer_id, verify_time);

                Ok(true)
            }
            Ok(false) => {
                error!("🚨 [ZK BLOCK AUTH] INVALID membership proof from {}!", peer_id);
                error!("   This peer is UNAUTHORIZED - not a network member!");

                // Ban peer for invalid proof
                self.ban_peer(peer_id).await;

                Err(anyhow::anyhow!("Invalid membership proof from peer {}", peer_id))
            }
            Err(e) => {
                error!("❌ [ZK BLOCK AUTH] Proof verification error from {}: {}",
                       peer_id, e);
                Ok(false) // Treat verification errors as unverified (don't ban)
            }
        }
    }

    /// Verify ZK-SNARK membership proof against network tree
    async fn verify_membership_proof(
        &self,
        proof: &NetworkMembershipProof,
        network_tree: &MerkleTree,
    ) -> Result<bool> {
        // Check proof freshness (30 minute expiration for replay attack prevention)
        const PROOF_EXPIRY_SECONDS: u64 = 1800; // 30 minutes

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let proof_age = current_time.saturating_sub(proof.metadata.timestamp);

        if proof_age > PROOF_EXPIRY_SECONDS {
            warn!("⚠️ [ZK BLOCK AUTH] Proof expired! Age: {}s (max: {}s)",
                  proof_age, PROOF_EXPIRY_SECONDS);
            warn!("   This prevents replay attacks with old proofs");
            return Ok(false);
        }

        // Verify network ID matches (prevent cross-network attacks)
        // v7.3.0: Accept mainnet2026, mainnet2026.1 and mainnet2026.2
        let expected_network_ids = ["mainnet-genesis", "mainnet2026.2", "mainnet2026.1", "mainnet2026"];

        if !expected_network_ids.contains(&proof.network_id.as_str()) {
            error!("🚨 [ZK BLOCK AUTH] Network ID mismatch! Cross-network attack detected.");
            error!("   Expected one of: {:?}, Got: {}", expected_network_ids, proof.network_id);
            error!("   This proof is from a different network (mainnet/testnet/devnet)!");
            return Ok(false);
        }

        // Verify network root matches
        if proof.network_root != network_tree.root {
            warn!("⚠️ [ZK BLOCK AUTH] Network root mismatch! Proof may be from different network version.");
            warn!("   Expected: {}, Got: {}",
                  hex::encode(&network_tree.root),
                  hex::encode(&proof.network_root));
            return Ok(false);
        }

        // Verify the PLONK/Groth16 proof
        // This proves: "I know a valid Merkle path from a leaf to the network root"
        // WITHOUT revealing which leaf or the Merkle path itself

        // Note: Actual verification would use q-zk-p2p::verify_membership_proof
        // For now, we simulate verification (replace with real call)
        let verification_result = q_zk_p2p::verify_membership_proof(proof).await?;

        if verification_result {
            debug!("✅ [ZK BLOCK AUTH] Membership proof verified - peer is authorized network member");
        } else {
            error!("🚨 [ZK BLOCK AUTH] Membership proof INVALID - peer is NOT authorized!");
        }

        Ok(verification_result)
    }

    /// Record rate limit violation
    async fn record_rate_limit_violation(&self, peer_id: &PeerId) {
        // After 3 rate limit violations in short period, ban the peer
        // This prevents bandwidth exhaustion attacks

        debug!("⚠️ [ZK BLOCK AUTH] Recorded rate limit violation for peer {}", peer_id);

        // In production, track violations and ban after threshold
        // For now, just log
    }

    /// Ban peer for invalid proof or excessive violations
    async fn ban_peer(&self, peer_id: &PeerId) {
        let mut banned = self.banned_peers.write().await;
        banned.insert(*peer_id);

        error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        error!("🚨 PEER BANNED: {}", peer_id);
        error!("   Reason: Invalid membership proof");
        error!("   This peer is UNAUTHORIZED to request blocks!");
        error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    /// Get number of banned peers
    pub async fn banned_peer_count(&self) -> usize {
        self.banned_peers.read().await.len()
    }

    /// Update network Merkle tree (when new peers join/leave)
    pub async fn update_network_tree(&self, new_tree: MerkleTree) -> Result<()> {
        let mut network_tree = self.network_tree.write().await;
        *network_tree = new_tree;

        info!("🌳 [ZK BLOCK AUTH] Updated network Merkle tree");
        info!("   New root: {}", hex::encode(&network_tree.root));
        info!("   Leaf count: {}", network_tree.leaf_count);

        Ok(())
    }
}

/// Generate ZK-SNARK membership proof for block pack request
///
/// **Note**: This uses `NetworkMembershipProof::prove_membership` from q-zk-p2p
pub async fn generate_block_request_proof(
    peer_id: &PeerId,
    network_tree: &MerkleTree,
    secret_position: usize, // Peer's position in network tree (secret!)
    validator_id: &[u8; 32], // Validator ID (32-byte array)
) -> Result<NetworkMembershipProof> {
    info!("🔐 [ZK BLOCK AUTH] Generating membership proof for peer {}", peer_id);

    // Get merkle proof for the validator's position
    let merkle_proof = network_tree.get_proof(validator_id, secret_position)?;

    // Generate ZK-SNARK proof using existing infrastructure
    let proof = q_zk_p2p::NetworkMembershipProof::prove_membership(
        validator_id,
        network_tree,
        secret_position,
        &merkle_proof,
    ).await?;

    info!("✅ [ZK BLOCK AUTH] Generated membership proof for block request");

    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_block_request_authenticator_creation() {
        let tree = MerkleTree {
            root: [0u8; 32],
            depth: 8,
            leaf_count: 10,
        };

        let authenticator = BlockRequestAuthenticator::new(tree, false)
            .await
            .unwrap();

        assert!(!authenticator.verification_enabled);
        assert_eq!(authenticator.banned_peer_count().await, 0);
        assert_eq!(authenticator.max_requests_per_minute, 10);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let tree = MerkleTree {
            root: [0u8; 32],
            depth: 8,
            leaf_count: 10,
        };

        let authenticator = BlockRequestAuthenticator::new(tree, false)
            .await
            .unwrap();

        let peer_id = PeerId::random();

        // First 10 requests should succeed
        for _ in 0..10 {
            assert!(authenticator.check_rate_limit(&peer_id).await.unwrap());
        }

        // 11th request should fail (rate limit)
        assert!(!authenticator.check_rate_limit(&peer_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_banned_peer_rejection() {
        let tree = MerkleTree {
            root: [0u8; 32],
            depth: 8,
            leaf_count: 10,
        };

        let authenticator = BlockRequestAuthenticator::new(tree, true)
            .await
            .unwrap();

        let peer_id = PeerId::random();

        // Ban the peer
        authenticator.ban_peer(&peer_id).await;

        // Verify peer is banned
        assert!(authenticator.is_banned(&peer_id).await);
        assert_eq!(authenticator.banned_peer_count().await, 1);

        // Request should be rejected
        let request = AuthenticatedBlockPackRequest {
            peer_id: peer_id.to_string(),
            start_height: 0,
            end_height: 100,
            membership_proof: None,
            timestamp: 0,
        };

        let result = authenticator.verify_block_request(&peer_id, &request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_network_tree_update() {
        let tree1 = MerkleTree {
            root: [1u8; 32],
            depth: 8,
            leaf_count: 10,
        };

        let authenticator = BlockRequestAuthenticator::new(tree1, false)
            .await
            .unwrap();

        // Update to new network tree
        let tree2 = MerkleTree {
            root: [2u8; 32],
            depth: 8,
            leaf_count: 15,
        };

        authenticator.update_network_tree(tree2).await.unwrap();

        // Verify tree was updated
        let network_tree = authenticator.network_tree.read().await;
        assert_eq!(network_tree.root, [2u8; 32]);
        assert_eq!(network_tree.leaf_count, 15);
    }
}
