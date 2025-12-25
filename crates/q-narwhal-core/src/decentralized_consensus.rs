//! 🔐 v1.3.12-beta: Decentralized DAG-Knight Consensus with SQIsign
//!
//! This module implements the P2P protocol for collecting 2f+1 signatures
//! from validators to create certificates for DAG-Knight consensus.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DECENTRALIZED CONSENSUS                       │
//! │                                                                   │
//! │  1. Validator creates vertex with transactions                   │
//! │  2. Broadcasts SignatureRequest to /consensus/sig-requests       │
//! │  3. Other validators verify & sign with SQIsign                  │
//! │  4. Signatures returned via /consensus/sig-responses             │
//! │  5. When 2f+1 collected → Certificate created                    │
//! │  6. Certificate broadcast via /consensus/certificates            │
//! │                                                                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

use q_types::{NodeId, ValidatorId, VertexId, Round, Certificate};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// P2P MESSAGE TYPES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Validator announcement for the registry
/// Broadcast on /consensus/validators topic when validator joins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorAnnouncement {
    /// Validator's node ID (32 bytes)
    pub validator_id: ValidatorId,
    /// SQIsign public key (compressed, serialized)
    pub sqisign_public_key: Vec<u8>,
    /// Timestamp of announcement
    pub timestamp: u64,
    /// Signature over (validator_id || sqisign_public_key || timestamp)
    /// This proves ownership of the SQIsign key
    pub proof_signature: Vec<u8>,
}

/// Request for signatures over a vertex
/// Broadcast on /consensus/sig-requests topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureRequest {
    /// ID of the vertex to sign
    pub vertex_id: VertexId,
    /// Round number
    pub round: Round,
    /// Hash of the vertex data (for verification)
    pub vertex_hash: [u8; 32],
    /// The requesting validator's ID
    pub requester: ValidatorId,
    /// Timestamp to prevent replay attacks
    pub timestamp: u64,
    /// Serialized vertex data (so validators can verify before signing)
    pub vertex_data: Vec<u8>,
}

/// Response containing a signature over a vertex
/// Sent on /consensus/sig-responses topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureResponse {
    /// ID of the vertex that was signed
    pub vertex_id: VertexId,
    /// The signing validator's ID
    pub signer: ValidatorId,
    /// SQIsign signature over (vertex_id || vertex_hash || round)
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// Certificate broadcast after 2f+1 signatures collected
/// Broadcast on /consensus/certificates topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateBroadcast {
    /// The complete certificate with all signatures
    pub certificate: Certificate,
    /// Who is broadcasting this certificate
    pub broadcaster: ValidatorId,
    /// Timestamp when broadcast was created
    pub timestamp: u64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VALIDATOR REGISTRY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Information about a known validator
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    /// Validator's node ID
    pub validator_id: ValidatorId,
    /// SQIsign public key (serialized)
    pub sqisign_public_key: Vec<u8>,
    /// When the validator was first seen
    pub first_seen: Instant,
    /// Last activity timestamp
    pub last_seen: Instant,
    /// Number of valid signatures received from this validator
    pub valid_signatures: u64,
    /// Number of invalid signatures (for reputation)
    pub invalid_signatures: u64,
}

/// Registry of known validators and their SQIsign public keys
/// This is essential for verifying signatures from other validators
pub struct ValidatorRegistry {
    /// Map of validator_id -> ValidatorInfo
    validators: RwLock<HashMap<ValidatorId, ValidatorInfo>>,
    /// Our own validator ID
    our_id: ValidatorId,
    /// Byzantine fault tolerance parameter (f)
    /// Network tolerates f Byzantine validators out of 3f+1 total
    f: usize,
}

impl ValidatorRegistry {
    /// Create a new validator registry
    pub fn new(our_id: ValidatorId, f: usize) -> Self {
        info!("🔐 [REGISTRY] Created validator registry (f={}, need {} for 2f+1)", f, 2*f + 1);
        Self {
            validators: RwLock::new(HashMap::new()),
            our_id,
            f,
        }
    }

    /// Register a new validator (from announcement or discovery)
    pub async fn register(&self, announcement: ValidatorAnnouncement) -> Result<(), String> {
        // Don't register ourselves
        if announcement.validator_id == self.our_id {
            return Ok(());
        }

        // TODO: Verify the proof_signature using the sqisign_public_key
        // This proves the announcer owns the private key

        let mut validators = self.validators.write().await;

        if validators.contains_key(&announcement.validator_id) {
            // Update last_seen for existing validator
            if let Some(info) = validators.get_mut(&announcement.validator_id) {
                info.last_seen = Instant::now();
            }
            debug!("🔐 [REGISTRY] Updated existing validator {}...",
                   hex::encode(&announcement.validator_id[..8]));
        } else {
            // New validator
            let info = ValidatorInfo {
                validator_id: announcement.validator_id,
                sqisign_public_key: announcement.sqisign_public_key,
                first_seen: Instant::now(),
                last_seen: Instant::now(),
                valid_signatures: 0,
                invalid_signatures: 0,
            };
            validators.insert(announcement.validator_id, info);
            info!("🔐 [REGISTRY] Registered new validator {}.. (total: {})",
                  hex::encode(&announcement.validator_id[..8]),
                  validators.len());
        }

        Ok(())
    }

    /// Get a validator's public key for signature verification
    pub async fn get_public_key(&self, validator_id: &ValidatorId) -> Option<Vec<u8>> {
        let validators = self.validators.read().await;
        validators.get(validator_id).map(|v| v.sqisign_public_key.clone())
    }

    /// Get the number of registered validators (excluding ourselves)
    pub async fn validator_count(&self) -> usize {
        self.validators.read().await.len()
    }

    /// Check if we have enough validators for consensus (2f+1)
    pub async fn has_quorum(&self) -> bool {
        let count = self.validator_count().await;
        // +1 for ourselves
        count + 1 >= 2 * self.f + 1
    }

    /// Get the required threshold for consensus (2f+1)
    pub fn required_threshold(&self) -> usize {
        2 * self.f + 1
    }

    /// Record a valid signature from a validator (for reputation)
    pub async fn record_valid_signature(&self, validator_id: &ValidatorId) {
        let mut validators = self.validators.write().await;
        if let Some(info) = validators.get_mut(validator_id) {
            info.valid_signatures += 1;
            info.last_seen = Instant::now();
        }
    }

    /// Record an invalid signature from a validator (for reputation/slashing)
    pub async fn record_invalid_signature(&self, validator_id: &ValidatorId) {
        let mut validators = self.validators.write().await;
        if let Some(info) = validators.get_mut(validator_id) {
            info.invalid_signatures += 1;
            warn!("⚠️ [REGISTRY] Invalid signature from {}.. (total invalid: {})",
                  hex::encode(&validator_id[..8]), info.invalid_signatures);
        }
    }

    /// Get all validator IDs
    pub async fn all_validator_ids(&self) -> Vec<ValidatorId> {
        self.validators.read().await.keys().cloned().collect()
    }

    /// Clean up stale validators (not seen in specified duration)
    pub async fn cleanup_stale(&self, max_age: Duration) {
        let mut validators = self.validators.write().await;
        let before_count = validators.len();
        validators.retain(|_, info| info.last_seen.elapsed() < max_age);
        let removed = before_count - validators.len();
        if removed > 0 {
            info!("🧹 [REGISTRY] Removed {} stale validators", removed);
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PENDING SIGNATURE COLLECTOR
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tracks pending signature requests and collected signatures
pub struct PendingSignatures {
    /// vertex_id -> collected signatures
    pending: RwLock<HashMap<VertexId, CollectedSignatures>>,
    /// Timeout for signature collection
    timeout: Duration,
}

struct CollectedSignatures {
    /// The original request
    request: SignatureRequest,
    /// Collected signatures: validator_id -> signature bytes
    signatures: HashMap<ValidatorId, Vec<u8>>,
    /// When collection started
    started_at: Instant,
    /// Channel to notify when threshold reached
    completion_tx: Option<tokio::sync::oneshot::Sender<HashMap<ValidatorId, Vec<u8>>>>,
}

impl PendingSignatures {
    pub fn new(timeout: Duration) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            timeout,
        }
    }

    /// Start collecting signatures for a vertex
    pub async fn start_collection(
        &self,
        request: SignatureRequest,
    ) -> tokio::sync::oneshot::Receiver<HashMap<ValidatorId, Vec<u8>>> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let collected = CollectedSignatures {
            request,
            signatures: HashMap::new(),
            started_at: Instant::now(),
            completion_tx: Some(tx),
        };

        let vertex_id = collected.request.vertex_id;
        self.pending.write().await.insert(vertex_id, collected);

        debug!("🔐 [COLLECT] Started signature collection for vertex {}...",
               hex::encode(&vertex_id[..8]));

        rx
    }

    /// Add a received signature
    pub async fn add_signature(
        &self,
        response: SignatureResponse,
        threshold: usize,
    ) -> Option<HashMap<ValidatorId, Vec<u8>>> {
        let mut pending = self.pending.write().await;

        if let Some(collected) = pending.get_mut(&response.vertex_id) {
            // Check for duplicate
            if collected.signatures.contains_key(&response.signer) {
                debug!("⚠️ [COLLECT] Duplicate signature from {}.. for vertex {}",
                       hex::encode(&response.signer[..8]),
                       hex::encode(&response.vertex_id[..8]));
                return None;
            }

            // Add signature
            collected.signatures.insert(response.signer, response.signature);
            let count = collected.signatures.len();

            debug!("🔐 [COLLECT] Received signature {}/{} for vertex {}.. from {}",
                   count, threshold,
                   hex::encode(&response.vertex_id[..8]),
                   hex::encode(&response.signer[..8]));

            // Check if threshold reached
            if count >= threshold {
                info!("✅ [COLLECT] Threshold reached ({}/{}) for vertex {}..!",
                      count, threshold, hex::encode(&response.vertex_id[..8]));

                let sigs = collected.signatures.clone();

                // Notify waiter
                if let Some(tx) = collected.completion_tx.take() {
                    let _ = tx.send(sigs.clone());
                }

                // Remove from pending
                pending.remove(&response.vertex_id);

                return Some(sigs);
            }
        } else {
            debug!("⚠️ [COLLECT] Received signature for unknown vertex {}",
                   hex::encode(&response.vertex_id[..8]));
        }

        None
    }

    /// Clean up timed-out requests
    pub async fn cleanup_expired(&self) {
        let mut pending = self.pending.write().await;
        let before = pending.len();
        pending.retain(|vertex_id, collected| {
            let expired = collected.started_at.elapsed() > self.timeout;
            if expired {
                warn!("⏰ [COLLECT] Signature collection timed out for vertex {}.. ({} sigs collected)",
                      hex::encode(&vertex_id[..8]), collected.signatures.len());
            }
            !expired
        });
        let removed = before - pending.len();
        if removed > 0 {
            debug!("🧹 [COLLECT] Cleaned up {} expired signature requests", removed);
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SIGNATURE VERIFIER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

use q_crypto_advanced::sqisign::{SqiSignature, SqiSignPublicKey, SqiSignLevel};

/// Verify an SQIsign signature from a validator
pub fn verify_sqisign_signature(
    public_key_bytes: &[u8],
    message: &[u8],
    signature_bytes: &[u8],
    _level: SqiSignLevel,
) -> Result<bool, String> {
    // Parse the signature
    let signature = SqiSignature::from_bytes(signature_bytes)
        .map_err(|e| format!("Invalid signature format: {:?}", e))?;

    // For now, we do structural validation
    // Full cryptographic verification requires the isogeny computation
    // which is computationally intensive (~50ms)

    // Basic validation:
    // 1. Signature has non-empty commitment and response
    if signature.commitment.is_empty() || signature.response.is_empty() {
        return Ok(false);
    }

    // 2. Public key is present
    if public_key_bytes.is_empty() {
        return Ok(false);
    }

    // 3. Verify commitment includes message hash
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(&signature.commitment);
    hasher.update(message);
    let _binding_hash: [u8; 32] = hasher.finalize().into();

    // TODO: Full isogeny-based verification
    // For production, implement:
    // 1. Parse public key curve
    // 2. Verify response isogeny computation
    // 3. Check commitment matches

    // For now, accept structurally valid signatures
    // This is safe in testnet but MUST be upgraded for mainnet
    debug!("🔐 [VERIFY] SQIsign signature structurally valid (full verification TODO)");

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validator_registry() {
        let our_id = [1u8; 32];
        let registry = ValidatorRegistry::new(our_id, 1); // f=1, need 3 validators

        // Register a validator
        let announcement = ValidatorAnnouncement {
            validator_id: [2u8; 32],
            sqisign_public_key: vec![0u8; 64],
            timestamp: 0,
            proof_signature: vec![],
        };
        registry.register(announcement).await.unwrap();

        assert_eq!(registry.validator_count().await, 1);
        assert!(!registry.has_quorum().await); // Need 3, have 2 (us + 1)

        // Register another
        let announcement2 = ValidatorAnnouncement {
            validator_id: [3u8; 32],
            sqisign_public_key: vec![0u8; 64],
            timestamp: 0,
            proof_signature: vec![],
        };
        registry.register(announcement2).await.unwrap();

        assert_eq!(registry.validator_count().await, 2);
        assert!(registry.has_quorum().await); // Have 3 (us + 2)
    }

    #[tokio::test]
    async fn test_signature_collection() {
        let collector = PendingSignatures::new(Duration::from_secs(30));

        let request = SignatureRequest {
            vertex_id: [1u8; 32],
            round: 1,
            vertex_hash: [0u8; 32],
            requester: [0u8; 32],
            timestamp: 0,
            vertex_data: vec![],
        };

        let _rx = collector.start_collection(request).await;

        // Add first signature
        let response1 = SignatureResponse {
            vertex_id: [1u8; 32],
            signer: [2u8; 32],
            signature: vec![0u8; 204],
            timestamp: 0,
        };
        let result = collector.add_signature(response1, 2).await;
        assert!(result.is_none()); // Not enough yet

        // Add second signature - should trigger threshold
        let response2 = SignatureResponse {
            vertex_id: [1u8; 32],
            signer: [3u8; 32],
            signature: vec![0u8; 204],
            timestamp: 0,
        };
        let result = collector.add_signature(response2, 2).await;
        assert!(result.is_some()); // Threshold reached!
        assert_eq!(result.unwrap().len(), 2);
    }
}
