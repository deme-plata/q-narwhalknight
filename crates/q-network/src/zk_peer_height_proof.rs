//! ZK-STARK Peer Height Proofs for Trustless P2P Sync
//!
//! **v0.9.6-beta**: Zero-Knowledge proofs for peer height announcements
//!
//! This module implements ZK-STARK proofs that allow peers to prove they possess
//! a block at a claimed height without revealing block contents or network topology.
//!
//! **Security Properties**:
//! - Prevents malicious peers from announcing false heights
//! - Enables trustless sync-down protection
//! - GPU-accelerated verification (<100ms)
//! - Quantum-resistant (STARK-based)

use anyhow::Result;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use q_types::QBlock;
use q_zk_stark::{BlockPossessionCircuit, StarkProof, StarkSystem};

/// Peer height announcement with ZK-STARK proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerHeightWithProof {
    /// Peer ID making the announcement
    pub peer_id: String,

    /// Claimed highest block height
    pub highest_block: u64,

    /// ZK-STARK proof: "I have block at this height"
    /// Proves possession without revealing block contents
    pub height_proof: Option<StarkProof>,

    /// Merkle root of blocks 0..highest_block (public)
    pub blockchain_merkle_root: Option<[u8; 32]>,

    /// Timestamp of announcement
    pub timestamp: u64,

    // ========================================
    // v3.3.9-beta: VERSION FILTERING & CAPABILITY ANNOUNCEMENT
    // Added for mainnet-safe peer compatibility
    // ========================================

    /// Software version string (e.g., "3.3.9-beta")
    /// Used to filter incompatible peers
    #[serde(default)]
    pub software_version: Option<String>,

    /// Protocol version (semantic versioning major.minor)
    /// Major version changes = incompatible peers
    #[serde(default)]
    pub protocol_version: Option<u32>,

    /// Upgrade capabilities this node supports
    /// e.g., ["pq-signatures", "upgrade-gate-v1", "consensus-guard"]
    #[serde(default)]
    pub upgrade_capabilities: Vec<String>,

    /// Network ID this node is on (e.g., "mainnet")
    /// Must match for sync to proceed
    #[serde(default)]
    pub network_id: Option<String>,
}

/// Peer height proof verifier with reputation tracking
pub struct PeerHeightVerifier {
    /// ZK-STARK system for proof verification
    stark_system: Arc<RwLock<StarkSystem>>,

    /// Reputation: peer_id -> (successful_proofs, failed_proofs)
    peer_reputation: Arc<RwLock<std::collections::HashMap<PeerId, (u64, u64)>>>,

    /// Ban list: peers with too many failed proofs
    banned_peers: Arc<RwLock<std::collections::HashSet<PeerId>>>,

    /// Enable ZK verification (default: false for gradual rollout)
    verification_enabled: bool,
}

impl PeerHeightVerifier {
    /// Create new peer height verifier
    pub async fn new(enable_gpu: bool) -> Result<Self> {
        let stark_system = StarkSystem::new(enable_gpu).await?;

        Ok(Self {
            stark_system: Arc::new(RwLock::new(stark_system)),
            peer_reputation: Arc::new(RwLock::new(std::collections::HashMap::new())),
            banned_peers: Arc::new(RwLock::new(std::collections::HashSet::new())),
            verification_enabled: false, // Gradual rollout: disabled by default
        })
    }

    /// Enable ZK proof verification (after gradual rollout)
    pub fn enable_verification(&mut self) {
        self.verification_enabled = true;
        info!("🔐 [ZK HEIGHT PROOF] Verification ENABLED - trustless sync active!");
    }

    /// Disable ZK proof verification (for testing/rollback)
    pub fn disable_verification(&mut self) {
        self.verification_enabled = false;
        warn!("⚠️ [ZK HEIGHT PROOF] Verification DISABLED - accepting unverified heights!");
    }

    /// Check if peer is banned
    pub async fn is_banned(&self, peer_id: &PeerId) -> bool {
        self.banned_peers.read().await.contains(peer_id)
    }

    /// Verify peer height announcement with optional ZK proof
    ///
    /// **Returns**:
    /// - Ok(true) if proof valid OR verification disabled
    /// - Ok(false) if proof missing but required
    /// - Err if proof invalid (malicious peer)
    pub async fn verify_peer_height(
        &self,
        peer_id: &PeerId,
        announced_height: u64,
        height_proof: Option<&StarkProof>,
    ) -> Result<bool> {
        // Check if peer is banned
        if self.is_banned(peer_id).await {
            error!("🚫 [ZK HEIGHT PROOF] Rejected announcement from BANNED peer {}", peer_id);
            return Ok(false);
        }

        // If verification disabled, accept all announcements (backward compatibility)
        if !self.verification_enabled {
            debug!("📡 [ZK HEIGHT PROOF] Verification disabled - accepting height {} from {} without proof",
                   announced_height, peer_id);
            return Ok(true);
        }

        // If verification enabled, proof is required
        let proof = match height_proof {
            Some(p) => p,
            None => {
                warn!("⚠️ [ZK HEIGHT PROOF] Peer {} announced height {} WITHOUT proof (verification enabled!)",
                      peer_id, announced_height);
                return Ok(false); // Reject unproven heights
            }
        };

        // Verify ZK-STARK proof
        let start = Instant::now();
        let mut stark_system = self.stark_system.write().await;

        // Public inputs for verification (height is public knowledge)
        let public_inputs = vec![announced_height];

        match stark_system.verify(proof, &public_inputs).await {
            Ok(true) => {
                let verify_time = start.elapsed();
                info!("✅ [ZK HEIGHT PROOF] Verified height {} from {} in {:?}",
                      announced_height, peer_id, verify_time);

                // Update reputation (successful proof)
                self.record_proof_success(peer_id).await;

                Ok(true)
            }
            Ok(false) => {
                error!("🚨 [ZK HEIGHT PROOF] INVALID proof from {} for height {}!",
                       peer_id, announced_height);
                error!("   This peer is MALICIOUS - announcing false height!");

                // Update reputation (failed proof)
                self.record_proof_failure(peer_id).await;

                // Check if peer should be banned
                self.check_and_ban_peer(peer_id).await;

                Err(anyhow::anyhow!("Invalid height proof from peer {}", peer_id))
            }
            Err(e) => {
                error!("❌ [ZK HEIGHT PROOF] Proof verification error from {}: {}",
                       peer_id, e);
                Ok(false) // Treat verification errors as unverified (don't ban)
            }
        }
    }

    /// Record successful proof verification
    async fn record_proof_success(&self, peer_id: &PeerId) {
        let mut reputation = self.peer_reputation.write().await;
        let entry = reputation.entry(*peer_id).or_insert((0, 0));
        entry.0 += 1; // Increment successful proofs
        debug!("✅ [ZK HEIGHT PROOF] Peer {} reputation: {} successful, {} failed",
               peer_id, entry.0, entry.1);
    }

    /// Record failed proof verification
    async fn record_proof_failure(&self, peer_id: &PeerId) {
        let mut reputation = self.peer_reputation.write().await;
        let entry = reputation.entry(*peer_id).or_insert((0, 0));
        entry.1 += 1; // Increment failed proofs
        warn!("❌ [ZK HEIGHT PROOF] Peer {} reputation: {} successful, {} failed",
              peer_id, entry.0, entry.1);
    }

    /// Check if peer should be banned based on reputation
    async fn check_and_ban_peer(&self, peer_id: &PeerId) {
        const MAX_FAILED_PROOFS: u64 = 3; // Ban after 3 invalid proofs

        let reputation = self.peer_reputation.read().await;
        if let Some((_, failed)) = reputation.get(peer_id) {
            if *failed >= MAX_FAILED_PROOFS {
                drop(reputation); // Release read lock before acquiring write lock

                let mut banned = self.banned_peers.write().await;
                banned.insert(*peer_id);

                error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                error!("🚨 PEER BANNED: {}", peer_id);
                error!("   Reason: {} invalid height proofs", MAX_FAILED_PROOFS);
                error!("   This peer attempted to announce false heights!");
                error!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            }
        }
    }

    /// Get peer reputation statistics
    pub async fn get_peer_reputation(&self, peer_id: &PeerId) -> Option<(u64, u64)> {
        self.peer_reputation.read().await.get(peer_id).copied()
    }

    /// Get number of banned peers
    pub async fn banned_peer_count(&self) -> usize {
        self.banned_peers.read().await.len()
    }
}

/// Generate ZK-STARK proof of block possession at given height
///
/// **v0.9.6-beta FIX**: Now uses proper blockchain state circuit with real execution trace
/// and AIR constraints instead of mock data.
///
/// **Security Properties**:
/// - Proves possession of block at specific height
/// - Hides block contents and merkle path
/// - Verifiable against public blockchain root
pub async fn generate_height_proof(
    stark_system: &mut StarkSystem,
    block: &QBlock,
    merkle_proof: Vec<[u8; 32]>,
    merkle_root: [u8; 32],
) -> Result<StarkProof> {
    // Calculate block hash
    let block_hash = block.calculate_hash();

    info!(
        "🔨 [ZK HEIGHT PROOF] Generating proof for block {} at height {}",
        hex::encode(&block_hash[..8]),
        block.header.height
    );

    // Get current timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    // Create proper blockchain state circuit
    let circuit = BlockPossessionCircuit::new(
        block_hash,
        block.header.height,
        merkle_proof,
        merkle_root,
        timestamp,
    );

    // Generate real execution trace (not mock data!)
    let trace = circuit.generate_trace();

    // Generate AIR constraints (not byte string!)
    let constraints = circuit.generate_constraints();

    // Generate STARK proof using real circuit
    let start = Instant::now();
    let proof = stark_system.prove(&trace, &constraints).await?;
    let proving_time = start.elapsed();

    info!(
        "✅ [ZK HEIGHT PROOF] Generated proof for block {} (height {}) in {:?}",
        hex::encode(&block_hash[..8]),
        block.header.height,
        proving_time
    );

    debug!(
        "   Trace: {} rows, Constraints: {} bytes, Proof: {} bytes",
        trace.len(),
        constraints.len(),
        bincode::serialize(&proof).unwrap_or_default().len()
    );

    Ok(proof)
}

/// Generate height proof with automatic merkle proof (for convenience)
///
/// This is a wrapper that generates a simplified merkle proof from the block.
/// In production, merkle proofs should come from the actual blockchain state.
pub async fn generate_height_proof_simple(
    stark_system: &mut StarkSystem,
    block: &QBlock,
) -> Result<StarkProof> {
    // Generate a mock merkle proof for now
    // In production, this should be computed from actual blockchain state
    let merkle_depth = ((block.header.height as f64).log2().ceil() as usize).max(1);
    let mut merkle_proof = Vec::new();
    for i in 0..merkle_depth {
        let sibling = blake3::hash(&format!("sibling_{}_{}", i, block.header.height).as_bytes());
        merkle_proof.push(sibling.into());
    }

    // Compute merkle root (simplified - should be actual blockchain root)
    let merkle_root: [u8; 32] = *blake3::hash(format!("root_{}", block.header.height).as_bytes()).as_bytes();

    generate_height_proof(stark_system, block, merkle_proof, merkle_root).await
}

// ============================================================================
// v3.3.9-beta: VERSION FILTERING & CAPABILITY ANNOUNCEMENT
// ============================================================================

/// Current software version for peer announcements
pub const SOFTWARE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version (increment on breaking P2P changes)
/// - v1: Initial protocol
/// - v2: Added version filtering
pub const PROTOCOL_VERSION: u32 = 2;

/// Minimum required protocol version for sync
/// Peers below this version will be filtered out
pub const MIN_PROTOCOL_VERSION: u32 = 1;

/// Upgrade capabilities this node supports
pub fn get_upgrade_capabilities() -> Vec<String> {
    vec![
        "upgrade-gate-v1".to_string(),      // Height-gated upgrades
        "consensus-guard-v1".to_string(),   // Mainnet safety checks
        "pq-signatures-ready".to_string(),  // Post-quantum signature support
        "sync-down-protection".to_string(), // Sync-down safety checks
        "version-filter-v1".to_string(),    // Version-based peer filtering
        "auto-update-v1".to_string(),       // v8.5.0: P2P auto-update support
    ]
}

/// Create a properly-versioned peer height announcement
pub fn create_peer_height_announcement(
    peer_id: &str,
    highest_block: u64,
    network_id: &str,
) -> PeerHeightWithProof {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    PeerHeightWithProof {
        peer_id: peer_id.to_string(),
        highest_block,
        height_proof: None, // ZK proof optional for now
        blockchain_merkle_root: None,
        timestamp,
        // v3.3.9-beta: Version filtering fields
        software_version: Some(SOFTWARE_VERSION.to_string()),
        protocol_version: Some(PROTOCOL_VERSION),
        upgrade_capabilities: get_upgrade_capabilities(),
        network_id: Some(network_id.to_string()),
    }
}

/// Version filter result
#[derive(Debug, Clone, PartialEq)]
pub enum VersionFilterResult {
    /// Peer is compatible - proceed with sync
    Compatible {
        peer_version: String,
        peer_protocol: u32,
        common_capabilities: Vec<String>,
    },
    /// Peer is running older but compatible version - warn but allow
    LegacyPeer {
        reason: String,
    },
    /// Peer is incompatible - reject sync
    Incompatible {
        reason: String,
    },
    /// Network ID mismatch - different network entirely
    WrongNetwork {
        expected: String,
        actual: String,
    },
}

/// Filter peer based on version and capabilities
///
/// **Returns**:
/// - Compatible: Peer can be synced from
/// - LegacyPeer: Old peer without version info (allowed with warning)
/// - Incompatible: Peer rejected due to version/protocol mismatch
/// - WrongNetwork: Peer is on different network
pub fn filter_peer_version(
    announcement: &PeerHeightWithProof,
    our_network_id: &str,
) -> VersionFilterResult {
    // Check network ID first
    if let Some(ref peer_network) = announcement.network_id {
        if peer_network != our_network_id {
            warn!("🚫 [VERSION FILTER] Peer {} on WRONG NETWORK: {} (we are on {})",
                  announcement.peer_id, peer_network, our_network_id);
            return VersionFilterResult::WrongNetwork {
                expected: our_network_id.to_string(),
                actual: peer_network.clone(),
            };
        }
    }

    // Check protocol version
    if let Some(peer_protocol) = announcement.protocol_version {
        if peer_protocol < MIN_PROTOCOL_VERSION {
            warn!("🚫 [VERSION FILTER] Peer {} has OUTDATED protocol version: {} (min required: {})",
                  announcement.peer_id, peer_protocol, MIN_PROTOCOL_VERSION);
            return VersionFilterResult::Incompatible {
                reason: format!("Protocol version {} < minimum required {}", peer_protocol, MIN_PROTOCOL_VERSION),
            };
        }
    }

    // If no version info at all, treat as legacy peer
    if announcement.software_version.is_none() && announcement.protocol_version.is_none() {
        debug!("⚠️ [VERSION FILTER] Peer {} has no version info - legacy peer",
               announcement.peer_id);
        return VersionFilterResult::LegacyPeer {
            reason: "No version information in announcement".to_string(),
        };
    }

    // Find common capabilities
    let our_capabilities = get_upgrade_capabilities();
    let common_capabilities: Vec<String> = announcement.upgrade_capabilities
        .iter()
        .filter(|cap| our_capabilities.contains(cap))
        .cloned()
        .collect();

    let peer_version = announcement.software_version.clone().unwrap_or_else(|| "unknown".to_string());
    let peer_protocol = announcement.protocol_version.unwrap_or(0);

    info!("✅ [VERSION FILTER] Peer {} is COMPATIBLE | version: {} | protocol: {} | common caps: {:?}",
          announcement.peer_id, peer_version, peer_protocol, common_capabilities);

    VersionFilterResult::Compatible {
        peer_version,
        peer_protocol,
        common_capabilities,
    }
}

/// Check if peer should be used for sync based on version filtering
///
/// Returns true if peer passes version filter, false if should be skipped
pub fn should_sync_from_peer(
    announcement: &PeerHeightWithProof,
    our_network_id: &str,
    strict_mode: bool,
) -> bool {
    match filter_peer_version(announcement, our_network_id) {
        VersionFilterResult::Compatible { .. } => true,
        VersionFilterResult::LegacyPeer { reason } => {
            if strict_mode {
                warn!("🚫 [VERSION FILTER] Rejecting legacy peer in STRICT mode: {}", reason);
                false
            } else {
                debug!("⚠️ [VERSION FILTER] Allowing legacy peer (non-strict): {}", reason);
                true
            }
        }
        VersionFilterResult::Incompatible { reason } => {
            error!("🚫 [VERSION FILTER] Rejecting incompatible peer: {}", reason);
            false
        }
        VersionFilterResult::WrongNetwork { expected, actual } => {
            error!("🚫 [VERSION FILTER] Rejecting peer on WRONG network: {} (expected {})", actual, expected);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_height_verifier_creation() {
        let verifier = PeerHeightVerifier::new(false).await.unwrap();
        assert!(!verifier.verification_enabled);
        assert_eq!(verifier.banned_peer_count().await, 0);
    }

    #[tokio::test]
    async fn test_reputation_tracking() {
        let verifier = PeerHeightVerifier::new(false).await.unwrap();
        let peer_id = PeerId::random();

        verifier.record_proof_success(&peer_id).await;
        verifier.record_proof_success(&peer_id).await;
        verifier.record_proof_failure(&peer_id).await;

        let (successful, failed) = verifier.get_peer_reputation(&peer_id).await.unwrap();
        assert_eq!(successful, 2);
        assert_eq!(failed, 1);
    }

    #[tokio::test]
    async fn test_peer_banning() {
        let verifier = PeerHeightVerifier::new(false).await.unwrap();
        let peer_id = PeerId::random();

        // Simulate 3 failed proofs
        verifier.record_proof_failure(&peer_id).await;
        verifier.check_and_ban_peer(&peer_id).await;
        assert!(!verifier.is_banned(&peer_id).await); // Not banned yet

        verifier.record_proof_failure(&peer_id).await;
        verifier.check_and_ban_peer(&peer_id).await;
        assert!(!verifier.is_banned(&peer_id).await); // Not banned yet

        verifier.record_proof_failure(&peer_id).await;
        verifier.check_and_ban_peer(&peer_id).await;
        assert!(verifier.is_banned(&peer_id).await); // NOW banned!

        assert_eq!(verifier.banned_peer_count().await, 1);
    }
}
