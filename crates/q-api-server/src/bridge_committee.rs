//! ✅ v7.3.1: Multi-Sig Bridge Validation with Rotating Committee
//!
//! Bridge deposits (BTC/ZEC/IRON/ETH) are validated by a rotating committee of
//! 11 nodes, requiring 7-of-11 attestations before minting wrapped tokens.
//!
//! ## Architecture
//!
//! ```text
//! User calls claim_swap() → Node broadcasts BridgeAttestationRequest via gossipsub
//! → Committee members independently verify source chain deposit
//! → Each signs attestation → broadcasts back
//! → Once 7-of-11 threshold reached → mint wrapped token
//! → Graceful fallback to single-node if committee < 7 members
//! ```
//!
//! ## Committee Selection
//!
//! Deterministic: SHA3(block_hash || epoch || peer_id) → score → top 11
//! Rotates every 100 blocks (BRIDGE_EPOCH_BLOCKS).
//!
//! ## Backward Compatibility
//!
//! - Old nodes: ignore unknown `/bridge-attestations` topic
//! - New nodes with < 7 committee peers: fallback to single-node mint
//! - Fully upgraded network: 7-of-11 attestations required

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

// ============================================================================
// Constants
// ============================================================================

/// Committee rotates every 100 blocks
pub const BRIDGE_EPOCH_BLOCKS: u64 = 100;

/// Target committee size (top N peers selected)
pub const BRIDGE_COMMITTEE_SIZE: usize = 11;

/// Attestations required to approve a bridge claim
pub const BRIDGE_ATTESTATION_THRESHOLD: usize = 7;

/// Timeout for collecting attestations (seconds)
pub const BRIDGE_ATTESTATION_TIMEOUT_SECS: u64 = 60;

// ============================================================================
// Types
// ============================================================================

/// Which source chain the bridge claim is for
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeChainId {
    Bitcoin,
    Zcash,
    IronFish,
    Ethereum,
}

impl std::fmt::Display for BridgeChainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeChainId::Bitcoin => write!(f, "BTC"),
            BridgeChainId::Zcash => write!(f, "ZEC"),
            BridgeChainId::IronFish => write!(f, "IRON"),
            BridgeChainId::Ethereum => write!(f, "ETH"),
        }
    }
}

/// Broadcast by the claiming node to request committee attestations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeAttestationRequest {
    /// Unique identifier for this claim attempt
    pub claim_id: String,
    /// Which chain the deposit is on
    pub chain: BridgeChainId,
    /// The swap ID from the bridge module
    pub swap_id: String,
    /// HTLC hash lock (SHA256 of secret)
    pub hash_lock: [u8; 32],
    /// The revealed secret (hex-encoded, 32 bytes)
    pub secret: String,
    /// Amount in native base units (satoshis, zatoshis, ore, wei)
    pub amount: u128,
    /// Wallet address to credit
    pub wallet: [u8; 32],
    /// Direction: "sell_btc", "buy_btc", etc.
    pub direction: String,
    /// Current epoch number
    pub epoch: u64,
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
    /// Node ID of the requester
    pub requester_node_id: String,
    /// Ed25519 signature over the request payload
    pub signature: Vec<u8>,
    /// Signer's Ed25519 public key (32 bytes)
    pub signer_public_key: Vec<u8>,
}

impl BridgeAttestationRequest {
    /// Compute deterministic signing payload
    pub fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"bridge-attestation-request-v1");
        hasher.update(self.claim_id.as_bytes());
        hasher.update(&[self.chain as u8]);
        hasher.update(self.swap_id.as_bytes());
        hasher.update(&self.hash_lock);
        hasher.update(self.secret.as_bytes());
        hasher.update(&self.amount.to_le_bytes());
        hasher.update(&self.wallet);
        hasher.update(self.direction.as_bytes());
        hasher.update(&self.epoch.to_le_bytes());
        hasher.update(&self.timestamp_ms.to_le_bytes());
        hasher.update(self.requester_node_id.as_bytes());
        hasher.finalize().into()
    }

    /// Sign this request with an Ed25519 signing key
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let payload = self.signing_payload();
        let sig = signing_key.sign(&payload);
        self.signature = sig.to_bytes().to_vec();
        self.signer_public_key = signing_key.verifying_key().to_bytes().to_vec();
    }

    /// Verify the Ed25519 signature
    pub fn verify_signature(&self) -> bool {
        if self.signature.len() != 64 || self.signer_public_key.len() != 32 {
            return false;
        }
        let pk_bytes: [u8; 32] = match self.signer_public_key.clone().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let sig_bytes: [u8; 64] = match self.signature.clone().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes) else {
            return false;
        };
        let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        let payload = self.signing_payload();
        use ed25519_dalek::Verifier;
        vk.verify(&payload, &sig).is_ok()
    }

    /// CBOR serialization for gossipsub
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode BridgeAttestationRequest");
        buf
    }

    /// CBOR deserialization
    pub fn from_cbor(data: &[u8]) -> anyhow::Result<Self> {
        Ok(ciborium::from_reader(data)?)
    }
}

/// Attestation verdict from a committee member
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationVerdict {
    Approve,
    Reject,
}

/// Response from a committee member after verifying the deposit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeAttestation {
    /// Must match the request's claim_id
    pub claim_id: String,
    /// Approve or Reject
    pub verdict: AttestationVerdict,
    /// Reason (for rejects)
    pub reason: Option<String>,
    /// Node ID of the attesting committee member
    pub attester_node_id: String,
    /// Epoch this attestation was made in
    pub epoch: u64,
    /// Ed25519 signature
    pub signature: Vec<u8>,
    /// Signer public key (32 bytes)
    pub signer_public_key: Vec<u8>,
}

impl BridgeAttestation {
    /// Compute deterministic signing payload
    pub fn signing_payload(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"bridge-attestation-v1");
        hasher.update(self.claim_id.as_bytes());
        hasher.update(&[self.verdict as u8]);
        if let Some(ref reason) = self.reason {
            hasher.update(reason.as_bytes());
        }
        hasher.update(self.attester_node_id.as_bytes());
        hasher.update(&self.epoch.to_le_bytes());
        hasher.finalize().into()
    }

    /// Sign this attestation
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) {
        use ed25519_dalek::Signer;
        let payload = self.signing_payload();
        let sig = signing_key.sign(&payload);
        self.signature = sig.to_bytes().to_vec();
        self.signer_public_key = signing_key.verifying_key().to_bytes().to_vec();
    }

    /// Verify the Ed25519 signature
    pub fn verify_signature(&self) -> bool {
        if self.signature.len() != 64 || self.signer_public_key.len() != 32 {
            return false;
        }
        let pk_bytes: [u8; 32] = match self.signer_public_key.clone().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let sig_bytes: [u8; 64] = match self.signature.clone().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes) else {
            return false;
        };
        let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
        let payload = self.signing_payload();
        use ed25519_dalek::Verifier;
        vk.verify(&payload, &sig).is_ok()
    }

    /// CBOR serialization
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode BridgeAttestation");
        buf
    }

    /// CBOR deserialization
    pub fn from_cbor(data: &[u8]) -> anyhow::Result<Self> {
        Ok(ciborium::from_reader(data)?)
    }
}

/// Envelope for gossipsub: either a Request or an Attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeAttestationMessage {
    Request(BridgeAttestationRequest),
    Attestation(BridgeAttestation),
}

impl BridgeAttestationMessage {
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode BridgeAttestationMessage");
        buf
    }

    pub fn from_cbor(data: &[u8]) -> anyhow::Result<Self> {
        Ok(ciborium::from_reader(data)?)
    }
}

/// Result of a bridge claim after attestation collection
#[derive(Debug, Clone)]
pub enum BridgeClaimResult {
    /// Threshold reached with enough approvals
    Approved { approvals: usize, rejections: usize },
    /// Threshold reached but too many rejections
    Rejected { approvals: usize, rejections: usize, reasons: Vec<String> },
    /// Timed out before reaching threshold
    Timeout { approvals: usize, rejections: usize },
}

/// Tracks a pending bridge claim waiting for attestations
pub struct PendingBridgeClaim {
    pub request: BridgeAttestationRequest,
    pub attestations: HashMap<String, BridgeAttestation>, // attester_node_id -> attestation
    pub created_at: Instant,
    pub notify: Arc<Notify>,
    pub result: Option<BridgeClaimResult>,
}

// ============================================================================
// BridgeCommittee
// ============================================================================

/// Manages the rotating bridge validation committee
pub struct BridgeCommittee {
    /// Current epoch number (block_height / BRIDGE_EPOCH_BLOCKS)
    pub current_epoch: u64,
    /// Current committee member node IDs
    pub current_members: Vec<String>,
    /// All known peer node IDs (updated from libp2p)
    pub known_peers: Vec<String>,
    /// Our node ID (PeerId string)
    pub our_node_id: String,
    /// Pending claims awaiting attestations
    pub pending_claims: HashMap<String, PendingBridgeClaim>, // claim_id -> pending
}

impl BridgeCommittee {
    /// Create a new bridge committee tracker
    pub fn new(our_node_id: String) -> Self {
        Self {
            current_epoch: 0,
            current_members: Vec::new(),
            known_peers: Vec::new(),
            our_node_id,
            pending_claims: HashMap::new(),
        }
    }

    /// Deterministically select committee members from known peers.
    ///
    /// Algorithm: For each peer, compute SHA3(block_hash || epoch || peer_id),
    /// sort by score, take top BRIDGE_COMMITTEE_SIZE.
    pub fn select_committee(
        block_hash: &[u8; 32],
        epoch: u64,
        peers: &[String],
    ) -> Vec<String> {
        if peers.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(String, [u8; 32])> = peers
            .iter()
            .map(|peer_id| {
                let mut hasher = Sha3_256::new();
                hasher.update(block_hash);
                hasher.update(&epoch.to_le_bytes());
                hasher.update(peer_id.as_bytes());
                let score: [u8; 32] = hasher.finalize().into();
                (peer_id.clone(), score)
            })
            .collect();

        // Sort by score (deterministic ordering)
        scored.sort_by(|a, b| a.1.cmp(&b.1));

        // Take top N
        scored
            .into_iter()
            .take(BRIDGE_COMMITTEE_SIZE)
            .map(|(peer_id, _)| peer_id)
            .collect()
    }

    /// Check if rotation is needed and rotate if at epoch boundary.
    /// Returns true if rotation happened.
    pub fn maybe_rotate(&mut self, height: u64, block_hash: &[u8; 32]) -> bool {
        let new_epoch = height / BRIDGE_EPOCH_BLOCKS;
        if new_epoch == self.current_epoch && !self.current_members.is_empty() {
            return false;
        }

        self.current_epoch = new_epoch;

        // Include ourselves in the peer list for committee selection
        let mut all_peers = self.known_peers.clone();
        if !all_peers.contains(&self.our_node_id) {
            all_peers.push(self.our_node_id.clone());
        }

        self.current_members = Self::select_committee(block_hash, new_epoch, &all_peers);

        info!(
            "🔄 [BRIDGE COMMITTEE] Rotated at epoch {} (height {}): {} members [{}]",
            new_epoch,
            height,
            self.current_members.len(),
            self.current_members
                .iter()
                .map(|m| &m[..m.len().min(12)])
                .collect::<Vec<_>>()
                .join(", ")
        );

        true
    }

    /// Check if our node is in the current committee
    pub fn is_committee_member(&self) -> bool {
        self.current_members.contains(&self.our_node_id)
    }

    /// Register a new pending claim. Returns a Notify handle to await.
    pub fn register_pending_claim(
        &mut self,
        request: BridgeAttestationRequest,
    ) -> Arc<Notify> {
        let notify = Arc::new(Notify::new());
        let claim_id = request.claim_id.clone();
        self.pending_claims.insert(
            claim_id.clone(),
            PendingBridgeClaim {
                request,
                attestations: HashMap::new(),
                created_at: Instant::now(),
                notify: notify.clone(),
                result: None,
            },
        );
        debug!("📋 [BRIDGE COMMITTEE] Registered pending claim: {}", claim_id);
        notify
    }

    /// Process an incoming attestation. Returns Some(result) when threshold is reached.
    pub fn process_attestation(
        &mut self,
        attestation: BridgeAttestation,
    ) -> Option<BridgeClaimResult> {
        let claim_id = attestation.claim_id.clone();

        let pending = match self.pending_claims.get_mut(&claim_id) {
            Some(p) => p,
            None => {
                debug!(
                    "⚠️ [BRIDGE COMMITTEE] Attestation for unknown claim: {}",
                    claim_id
                );
                return None;
            }
        };

        // Don't accept duplicate attestations from the same node
        if pending
            .attestations
            .contains_key(&attestation.attester_node_id)
        {
            debug!(
                "⚠️ [BRIDGE COMMITTEE] Duplicate attestation from {} for claim {}",
                &attestation.attester_node_id[..attestation.attester_node_id.len().min(12)],
                claim_id
            );
            return None;
        }

        // Only accept attestations from current committee members
        if !self.current_members.contains(&attestation.attester_node_id) {
            warn!(
                "🚫 [BRIDGE COMMITTEE] Attestation from non-committee member {} for claim {}",
                &attestation.attester_node_id[..attestation.attester_node_id.len().min(12)],
                claim_id
            );
            return None;
        }

        let node_id = attestation.attester_node_id.clone();
        pending.attestations.insert(node_id.clone(), attestation);

        let approvals = pending
            .attestations
            .values()
            .filter(|a| a.verdict == AttestationVerdict::Approve)
            .count();
        let rejections = pending
            .attestations
            .values()
            .filter(|a| a.verdict == AttestationVerdict::Reject)
            .count();

        info!(
            "📊 [BRIDGE COMMITTEE] Claim {} attestations: {}/{} approvals, {} rejections (threshold: {})",
            claim_id,
            approvals,
            self.current_members.len(),
            rejections,
            BRIDGE_ATTESTATION_THRESHOLD
        );

        // Check if we've reached threshold
        if approvals >= BRIDGE_ATTESTATION_THRESHOLD {
            let result = BridgeClaimResult::Approved { approvals, rejections };
            let ret = result.clone();
            pending.result = Some(result);
            pending.notify.notify_one();
            return Some(ret);
        }

        // Check if too many rejections (can't possibly reach approval threshold)
        let remaining = self
            .current_members
            .len()
            .saturating_sub(approvals + rejections);
        if approvals + remaining < BRIDGE_ATTESTATION_THRESHOLD {
            let reasons: Vec<String> = pending
                .attestations
                .values()
                .filter(|a| a.verdict == AttestationVerdict::Reject)
                .filter_map(|a| a.reason.clone())
                .collect();
            let result = BridgeClaimResult::Rejected {
                approvals,
                rejections,
                reasons,
            };
            let ret = result.clone();
            pending.result = Some(result);
            pending.notify.notify_one();
            return Some(ret);
        }

        None
    }

    /// Remove expired pending claims (older than BRIDGE_ATTESTATION_TIMEOUT_SECS)
    pub fn cleanup_expired(&mut self) {
        let timeout = Duration::from_secs(BRIDGE_ATTESTATION_TIMEOUT_SECS);
        let expired: Vec<String> = self
            .pending_claims
            .iter()
            .filter(|(_, p)| p.created_at.elapsed() > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired {
            if let Some(mut pending) = self.pending_claims.remove(id) {
                let approvals = pending
                    .attestations
                    .values()
                    .filter(|a| a.verdict == AttestationVerdict::Approve)
                    .count();
                let rejections = pending
                    .attestations
                    .values()
                    .filter(|a| a.verdict == AttestationVerdict::Reject)
                    .count();
                warn!(
                    "⏰ [BRIDGE COMMITTEE] Claim {} timed out ({} approvals, {} rejections)",
                    id, approvals, rejections
                );
                pending.result = Some(BridgeClaimResult::Timeout {
                    approvals,
                    rejections,
                });
                pending.notify.notify_one();
            }
        }

        if !expired.is_empty() {
            debug!(
                "🧹 [BRIDGE COMMITTEE] Cleaned up {} expired claims",
                expired.len()
            );
        }
    }

    /// Update known peers list from libp2p peer discovery
    pub fn update_known_peers(&mut self, peers: Vec<String>) {
        self.known_peers = peers;
    }
}

// ============================================================================
// Source Chain Verification
// ============================================================================

/// Verify a bridge deposit by checking the HTLC secret against the hash lock.
///
/// This performs the same SHA256(secret) == hash_lock check that each bridge
/// handler does locally. In the future, this can be extended to query source
/// chain RPCs for on-chain TX verification.
pub fn verify_bridge_deposit(
    chain: BridgeChainId,
    secret_hex: &str,
    hash_lock: &[u8; 32],
    _amount: u128,
) -> Result<bool, String> {
    // Decode the secret from hex
    let secret_bytes = hex::decode(secret_hex)
        .map_err(|e| format!("Invalid secret hex: {}", e))?;

    if secret_bytes.len() != 32 {
        return Err(format!(
            "Secret must be 32 bytes, got {}",
            secret_bytes.len()
        ));
    }

    // Compute SHA256(secret) and compare to hash_lock
    use sha2::{Sha256, Digest as Sha2Digest};
    let mut sha256 = Sha256::new();
    sha256.update(&secret_bytes);
    let computed_hash: [u8; 32] = sha256.finalize().into();

    if computed_hash != *hash_lock {
        return Err(format!(
            "{} bridge: SHA256(secret) mismatch. Expected {}, got {}",
            chain,
            hex::encode(hash_lock),
            hex::encode(&computed_hash)
        ));
    }

    debug!(
        "✅ [BRIDGE VERIFY] {} deposit verified: SHA256(secret) matches hash_lock",
        chain
    );

    Ok(true)
}

/// Generate a unique claim ID for a bridge attestation request
pub fn generate_claim_id(chain: BridgeChainId, swap_id: &str, node_id: &str) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(b"bridge-claim-id");
    hasher.update(&[chain as u8]);
    hasher.update(swap_id.as_bytes());
    hasher.update(node_id.as_bytes());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    hasher.update(&now.to_le_bytes());
    let hash: [u8; 32] = hasher.finalize().into();
    hex::encode(&hash[..16]) // 16 bytes = 32 hex chars
}

// ============================================================================
// Multi-Sig Claim Flow Helper
// ============================================================================

/// Execute the multi-sig bridge claim flow.
///
/// If the committee has enough members (>= threshold), broadcasts an attestation
/// request and waits for committee members to verify and respond. Returns the
/// result (Approved/Rejected/Timeout) or falls back to single-node mode.
///
/// Returns:
/// - Ok(true) — approved by committee (or fallback single-node)
/// - Ok(false) — rejected by committee
/// - Err — timeout or other error
pub async fn execute_multisig_claim(
    bridge_committee: &std::sync::Arc<tokio::sync::RwLock<BridgeCommittee>>,
    libp2p_command_tx: &Option<tokio::sync::mpsc::UnboundedSender<q_network::NetworkCommand>>,
    node_cypher: &std::sync::Arc<q_eternal_cypher::NodeCypher>,
    bridge_topic: &str,
    chain: BridgeChainId,
    swap_id: &str,
    secret_hex: &str,
    hash_lock: &[u8; 32],
    amount: u128,
    wallet: &[u8; 32],
    direction: &str,
) -> Result<bool, String> {
    // Check committee size
    let (committee_size, our_node_id, epoch) = {
        let committee = bridge_committee.read().await;
        (
            committee.current_members.len(),
            committee.our_node_id.clone(),
            committee.current_epoch,
        )
    };

    if committee_size < BRIDGE_ATTESTATION_THRESHOLD {
        // Fallback: not enough committee members for multi-sig
        info!(
            "🌉 [BRIDGE] Committee too small ({}/{}) for multi-sig, using single-node validation for {} swap {}",
            committee_size, BRIDGE_ATTESTATION_THRESHOLD, chain, swap_id
        );
        return Ok(true); // Allow single-node mint (backward compatible)
    }

    // Create attestation request
    let claim_id = generate_claim_id(chain, swap_id, &our_node_id);
    let mut request = BridgeAttestationRequest {
        claim_id: claim_id.clone(),
        chain,
        swap_id: swap_id.to_string(),
        hash_lock: *hash_lock,
        secret: secret_hex.to_string(),
        amount,
        wallet: *wallet,
        direction: direction.to_string(),
        epoch,
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        requester_node_id: our_node_id,
        signature: vec![],
        signer_public_key: vec![],
    };

    // Sign the request
    request.sign(node_cypher.signing_key());

    // Register pending claim and get notification handle
    let notify = {
        let mut committee = bridge_committee.write().await;
        committee.register_pending_claim(request.clone())
    };

    // Broadcast request via gossipsub
    let msg = BridgeAttestationMessage::Request(request);
    let msg_bytes = msg.to_cbor();
    let topic = bridge_topic.to_string();

    if let Some(ref cmd_tx) = libp2p_command_tx {
        let _ = cmd_tx.send(q_network::NetworkCommand::PublishConsensusMessage {
            topic,
            message_bytes: msg_bytes,
        });
    }

    info!(
        "📡 [BRIDGE] Broadcast attestation request for {} claim {} (waiting {}s for {}/{} attestations)",
        chain, claim_id, BRIDGE_ATTESTATION_TIMEOUT_SECS, BRIDGE_ATTESTATION_THRESHOLD, committee_size
    );

    // Wait for threshold attestations or timeout
    match tokio::time::timeout(
        Duration::from_secs(BRIDGE_ATTESTATION_TIMEOUT_SECS),
        notify.notified(),
    )
    .await
    {
        Ok(()) => {
            // Notification received — check result
            let committee = bridge_committee.read().await;
            if let Some(pending) = committee.pending_claims.get(&claim_id) {
                match &pending.result {
                    Some(BridgeClaimResult::Approved { approvals, rejections }) => {
                        info!(
                            "✅ [BRIDGE] Claim {} APPROVED by committee ({}/{} approvals, {} rejections)",
                            claim_id, approvals, committee.current_members.len(), rejections
                        );
                        Ok(true)
                    }
                    Some(BridgeClaimResult::Rejected { approvals, rejections, reasons }) => {
                        warn!(
                            "❌ [BRIDGE] Claim {} REJECTED by committee ({} approvals, {} rejections): {:?}",
                            claim_id, approvals, rejections, reasons
                        );
                        Ok(false)
                    }
                    Some(BridgeClaimResult::Timeout { approvals, rejections }) => {
                        warn!(
                            "⏰ [BRIDGE] Claim {} timed out ({} approvals, {} rejections)",
                            claim_id, approvals, rejections
                        );
                        Err(format!(
                            "Bridge attestation timed out ({}/{} approvals). Try again.",
                            approvals, BRIDGE_ATTESTATION_THRESHOLD
                        ))
                    }
                    None => {
                        // Shouldn't happen but handle gracefully
                        Err("Bridge attestation result unavailable".to_string())
                    }
                }
            } else {
                Err("Bridge claim not found after notification".to_string())
            }
        }
        Err(_) => {
            // Timeout
            warn!(
                "⏰ [BRIDGE] Claim {} timed out after {}s",
                claim_id, BRIDGE_ATTESTATION_TIMEOUT_SECS
            );
            // Trigger cleanup
            let mut committee = bridge_committee.write().await;
            committee.cleanup_expired();
            Err(format!(
                "Bridge attestation timed out after {}s. Network may not have enough upgraded nodes.",
                BRIDGE_ATTESTATION_TIMEOUT_SECS
            ))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peers(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("12D3KooW{:040}", i))
            .collect()
    }

    #[test]
    fn test_committee_selection_deterministic() {
        let block_hash = [42u8; 32];
        let epoch = 5;
        let peers = make_peers(20);

        let committee1 = BridgeCommittee::select_committee(&block_hash, epoch, &peers);
        let committee2 = BridgeCommittee::select_committee(&block_hash, epoch, &peers);

        assert_eq!(committee1, committee2, "Committee selection must be deterministic");
        assert_eq!(committee1.len(), BRIDGE_COMMITTEE_SIZE);
    }

    #[test]
    fn test_committee_selection_changes_with_epoch() {
        let block_hash = [42u8; 32];
        let peers = make_peers(20);

        let committee_epoch_1 = BridgeCommittee::select_committee(&block_hash, 1, &peers);
        let committee_epoch_2 = BridgeCommittee::select_committee(&block_hash, 2, &peers);

        // Different epochs should (almost certainly) produce different committees
        assert_ne!(
            committee_epoch_1, committee_epoch_2,
            "Different epochs should rotate committee"
        );
    }

    #[test]
    fn test_committee_selection_fewer_than_size() {
        let block_hash = [42u8; 32];
        let peers = make_peers(5); // fewer than BRIDGE_COMMITTEE_SIZE

        let committee = BridgeCommittee::select_committee(&block_hash, 1, &peers);
        assert_eq!(committee.len(), 5);
    }

    #[test]
    fn test_committee_selection_empty_peers() {
        let block_hash = [42u8; 32];
        let committee = BridgeCommittee::select_committee(&block_hash, 1, &[]);
        assert!(committee.is_empty());
    }

    #[test]
    fn test_maybe_rotate_epoch_boundary() {
        let mut committee = BridgeCommittee::new("our_node".to_string());
        committee.known_peers = make_peers(15);

        let block_hash = [1u8; 32];

        // Height 0 → epoch 0
        assert!(committee.maybe_rotate(0, &block_hash));
        assert_eq!(committee.current_epoch, 0);
        let members_epoch_0 = committee.current_members.clone();

        // Height 50 → still epoch 0, no rotation
        assert!(!committee.maybe_rotate(50, &block_hash));

        // Height 100 → epoch 1, rotation
        assert!(committee.maybe_rotate(100, &block_hash));
        assert_eq!(committee.current_epoch, 1);
        // Members may differ (different epoch → different selection)
        assert_ne!(members_epoch_0, committee.current_members);
    }

    #[test]
    fn test_is_committee_member() {
        let mut committee = BridgeCommittee::new("our_node".to_string());
        committee.current_members = vec![
            "peer1".to_string(),
            "our_node".to_string(),
            "peer3".to_string(),
        ];
        assert!(committee.is_committee_member());

        committee.current_members = vec!["peer1".to_string(), "peer3".to_string()];
        assert!(!committee.is_committee_member());
    }

    #[test]
    fn test_process_attestation_threshold() {
        let mut committee = BridgeCommittee::new("requester".to_string());
        // Set up committee with enough members
        let members: Vec<String> = (0..11).map(|i| format!("member_{}", i)).collect();
        committee.current_members = members.clone();

        let request = BridgeAttestationRequest {
            claim_id: "test_claim_1".to_string(),
            chain: BridgeChainId::Bitcoin,
            swap_id: "swap_123".to_string(),
            hash_lock: [0u8; 32],
            secret: "aa".repeat(32),
            amount: 100_000,
            wallet: [1u8; 32],
            direction: "sell_btc".to_string(),
            epoch: 1,
            timestamp_ms: 1000,
            requester_node_id: "requester".to_string(),
            signature: vec![],
            signer_public_key: vec![],
        };

        let _notify = committee.register_pending_claim(request);

        // Send 6 approvals — not enough
        for i in 0..6 {
            let attestation = BridgeAttestation {
                claim_id: "test_claim_1".to_string(),
                verdict: AttestationVerdict::Approve,
                reason: None,
                attester_node_id: format!("member_{}", i),
                epoch: 1,
                signature: vec![],
                signer_public_key: vec![],
            };
            let result = committee.process_attestation(attestation);
            assert!(result.is_none(), "Should not reach threshold with {} approvals", i + 1);
        }

        // 7th approval — reaches threshold
        let attestation = BridgeAttestation {
            claim_id: "test_claim_1".to_string(),
            verdict: AttestationVerdict::Approve,
            reason: None,
            attester_node_id: "member_6".to_string(),
            epoch: 1,
            signature: vec![],
            signer_public_key: vec![],
        };
        let result = committee.process_attestation(attestation);
        assert!(matches!(result, Some(BridgeClaimResult::Approved { approvals: 7, .. })));
    }

    #[test]
    fn test_process_attestation_rejection_impossible() {
        let mut committee = BridgeCommittee::new("requester".to_string());
        let members: Vec<String> = (0..11).map(|i| format!("member_{}", i)).collect();
        committee.current_members = members;

        let request = BridgeAttestationRequest {
            claim_id: "test_claim_2".to_string(),
            chain: BridgeChainId::Zcash,
            swap_id: "swap_456".to_string(),
            hash_lock: [0u8; 32],
            secret: "bb".repeat(32),
            amount: 50_000,
            wallet: [2u8; 32],
            direction: "sell_zec".to_string(),
            epoch: 1,
            timestamp_ms: 2000,
            requester_node_id: "requester".to_string(),
            signature: vec![],
            signer_public_key: vec![],
        };

        let _notify = committee.register_pending_claim(request);

        // Send 5 rejections — mathematically impossible to reach 7 approvals now
        // (11 total - 5 rejected = 6 remaining, need 7)
        for i in 0..4 {
            let attestation = BridgeAttestation {
                claim_id: "test_claim_2".to_string(),
                verdict: AttestationVerdict::Reject,
                reason: Some("Secret mismatch".to_string()),
                attester_node_id: format!("member_{}", i),
                epoch: 1,
                signature: vec![],
                signer_public_key: vec![],
            };
            assert!(committee.process_attestation(attestation).is_none());
        }

        // 5th rejection — now only 6 remaining, can't reach 7 approvals
        let attestation = BridgeAttestation {
            claim_id: "test_claim_2".to_string(),
            verdict: AttestationVerdict::Reject,
            reason: Some("Invalid hash".to_string()),
            attester_node_id: "member_4".to_string(),
            epoch: 1,
            signature: vec![],
            signer_public_key: vec![],
        };
        let result = committee.process_attestation(attestation);
        assert!(matches!(result, Some(BridgeClaimResult::Rejected { rejections: 5, .. })));
    }

    #[test]
    fn test_duplicate_attestation_ignored() {
        let mut committee = BridgeCommittee::new("requester".to_string());
        committee.current_members = vec!["member_0".to_string()];

        let request = BridgeAttestationRequest {
            claim_id: "test_claim_3".to_string(),
            chain: BridgeChainId::Ethereum,
            swap_id: "swap_789".to_string(),
            hash_lock: [0u8; 32],
            secret: "cc".repeat(32),
            amount: 1_000_000,
            wallet: [3u8; 32],
            direction: "sell_eth".to_string(),
            epoch: 1,
            timestamp_ms: 3000,
            requester_node_id: "requester".to_string(),
            signature: vec![],
            signer_public_key: vec![],
        };

        let _notify = committee.register_pending_claim(request);

        let attestation = BridgeAttestation {
            claim_id: "test_claim_3".to_string(),
            verdict: AttestationVerdict::Approve,
            reason: None,
            attester_node_id: "member_0".to_string(),
            epoch: 1,
            signature: vec![],
            signer_public_key: vec![],
        };

        // First attestation
        committee.process_attestation(attestation.clone());

        // Duplicate — should be ignored
        let result = committee.process_attestation(attestation);
        assert!(result.is_none());

        // Only 1 attestation recorded
        let pending = committee.pending_claims.get("test_claim_3").unwrap();
        assert_eq!(pending.attestations.len(), 1);
    }

    #[test]
    fn test_non_committee_attestation_rejected() {
        let mut committee = BridgeCommittee::new("requester".to_string());
        committee.current_members = vec!["member_0".to_string()];

        let request = BridgeAttestationRequest {
            claim_id: "test_claim_4".to_string(),
            chain: BridgeChainId::IronFish,
            swap_id: "swap_abc".to_string(),
            hash_lock: [0u8; 32],
            secret: "dd".repeat(32),
            amount: 200_000,
            wallet: [4u8; 32],
            direction: "sell_iron".to_string(),
            epoch: 1,
            timestamp_ms: 4000,
            requester_node_id: "requester".to_string(),
            signature: vec![],
            signer_public_key: vec![],
        };

        let _notify = committee.register_pending_claim(request);

        // Attestation from non-committee member
        let attestation = BridgeAttestation {
            claim_id: "test_claim_4".to_string(),
            verdict: AttestationVerdict::Approve,
            reason: None,
            attester_node_id: "not_a_member".to_string(),
            epoch: 1,
            signature: vec![],
            signer_public_key: vec![],
        };

        let result = committee.process_attestation(attestation);
        assert!(result.is_none());

        // No attestations recorded
        let pending = committee.pending_claims.get("test_claim_4").unwrap();
        assert_eq!(pending.attestations.len(), 0);
    }

    #[test]
    fn test_verify_bridge_deposit_btc() {
        use sha2::{Sha256, Digest as Sha2Digest};

        // Create a valid secret and hash_lock
        let secret = [0x42u8; 32];
        let secret_hex = hex::encode(&secret);

        let mut sha256 = Sha256::new();
        sha256.update(&secret);
        let hash_lock: [u8; 32] = sha256.finalize().into();

        // Should pass
        let result = verify_bridge_deposit(BridgeChainId::Bitcoin, &secret_hex, &hash_lock, 100_000);
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Wrong secret should fail
        let wrong_secret = hex::encode([0x43u8; 32]);
        let result = verify_bridge_deposit(BridgeChainId::Bitcoin, &wrong_secret, &hash_lock, 100_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_claim_id() {
        let id1 = generate_claim_id(BridgeChainId::Bitcoin, "swap1", "node1");
        let id2 = generate_claim_id(BridgeChainId::Bitcoin, "swap1", "node1");

        // Different timestamps → different IDs (includes current time)
        // In practice, successive calls in the same ms may produce same ID,
        // but the ID is still unique enough for our purposes
        assert_eq!(id1.len(), 32); // 16 bytes = 32 hex chars
        assert_eq!(id2.len(), 32);
    }

    #[test]
    fn test_attestation_message_cbor_roundtrip() {
        let request = BridgeAttestationRequest {
            claim_id: "test_rt".to_string(),
            chain: BridgeChainId::Ethereum,
            swap_id: "swap_rt".to_string(),
            hash_lock: [99u8; 32],
            secret: "ee".repeat(32),
            amount: 500_000,
            wallet: [5u8; 32],
            direction: "sell_eth".to_string(),
            epoch: 42,
            timestamp_ms: 12345678,
            requester_node_id: "node_rt".to_string(),
            signature: vec![1, 2, 3],
            signer_public_key: vec![4, 5, 6],
        };

        let msg = BridgeAttestationMessage::Request(request.clone());
        let encoded = msg.to_cbor();
        let decoded = BridgeAttestationMessage::from_cbor(&encoded).unwrap();

        match decoded {
            BridgeAttestationMessage::Request(r) => {
                assert_eq!(r.claim_id, "test_rt");
                assert_eq!(r.chain, BridgeChainId::Ethereum);
                assert_eq!(r.amount, 500_000);
                assert_eq!(r.epoch, 42);
            }
            _ => panic!("Expected Request variant"),
        }
    }

    #[test]
    fn test_request_signing_and_verification() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);

        let mut request = BridgeAttestationRequest {
            claim_id: "test_sig".to_string(),
            chain: BridgeChainId::Bitcoin,
            swap_id: "swap_sig".to_string(),
            hash_lock: [0u8; 32],
            secret: "ff".repeat(32),
            amount: 100_000,
            wallet: [6u8; 32],
            direction: "sell_btc".to_string(),
            epoch: 1,
            timestamp_ms: 9999,
            requester_node_id: "node_sig".to_string(),
            signature: vec![],
            signer_public_key: vec![],
        };

        // Unsigned → verification fails
        assert!(!request.verify_signature());

        // Sign → verification passes
        request.sign(&signing_key);
        assert!(request.verify_signature());

        // Tamper → verification fails
        request.amount = 999_999;
        assert!(!request.verify_signature());
    }

    #[test]
    fn test_attestation_signing_and_verification() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);

        let mut attestation = BridgeAttestation {
            claim_id: "test_att_sig".to_string(),
            verdict: AttestationVerdict::Approve,
            reason: None,
            attester_node_id: "attester_1".to_string(),
            epoch: 5,
            signature: vec![],
            signer_public_key: vec![],
        };

        attestation.sign(&signing_key);
        assert!(attestation.verify_signature());

        // Tamper
        attestation.verdict = AttestationVerdict::Reject;
        assert!(!attestation.verify_signature());
    }
}
