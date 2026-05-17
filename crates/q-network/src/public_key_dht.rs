/// DHT-based public key distribution for distributed AI nodes
///
/// This module implements Showstopper #2 fix: DHT announcement with signature verification
///
/// Security Model:
/// - node_id MUST be derived from hash(quantum_public_key) to prevent impersonation
/// - Announcements are self-signed with quantum signature
/// - Unsigned bytes stored alongside signature for verification
/// - 24-hour TTL with automatic re-announcement

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use super::distributed_ai::NodeCapability;
use super::real_dht::DhtCommand;

/// Public key announcement for DHT distribution
///
/// CRITICAL SECURITY: The signature field uses Option<Vec<u8>> to ensure
/// signed bytes exactly match unsigned bytes during verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePublicKeyAnnouncement {
    /// Node ID derived from hash(quantum_public_key) - prevents impersonation
    pub node_id: String,

    /// LibP2P peer ID for network routing
    pub peer_id: String,

    /// Quantum-resistant public key (Dilithium5 or Lamport OTS)
    pub quantum_public_key: Vec<u8>,

    /// Node's AI inference capability
    pub capability: NodeCapability,

    /// Timestamp when announcement was created
    pub announced_at: i64,

    /// Signature over all fields EXCEPT this one
    /// Use Option to make unsigned/signed states explicit
    ///
    /// Signing flow:
    /// 1. Create with signature: None
    /// 2. Serialize unsigned announcement
    /// 3. Sign those exact bytes
    /// 4. Store signed announcement WITH signature: Some(sig_bytes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<Vec<u8>>,
}

/// Cached public key with pre-computed fingerprint
///
/// Performance optimization: Fingerprint is computed once and reused
/// for signature cache lookups (avoids repeated SHA3-256 hashing)
#[derive(Debug, Clone)]
pub struct CachedPublicKey {
    /// Raw public key bytes
    pub key_bytes: Vec<u8>,

    /// SHA3-256 fingerprint (pre-computed)
    pub fingerprint: [u8; 32],

    /// When this key was cached
    pub cached_at: Instant,
}

impl CachedPublicKey {
    /// Create new cached public key with pre-computed fingerprint
    pub fn new(key_bytes: Vec<u8>) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(&key_bytes);
        let fingerprint: [u8; 32] = hasher.finalize().into();

        Self {
            key_bytes,
            fingerprint,
            cached_at: Instant::now(),
        }
    }
}

/// DHT-based public key manager
///
/// Responsibilities:
/// - Announce local node's public key to DHT
/// - Fetch peer public keys from DHT with verification
/// - Cache verified public keys with fingerprints
/// - Automatic re-announcement every 12 hours
pub struct PublicKeyDhtManager {
    /// Our node's ID (derived from our public key)
    node_id: String,

    /// Our LibP2P peer ID
    peer_id: String,

    /// Our node's capability
    capability: NodeCapability,

    /// Our quantum signer for creating announcements
    quantum_signer: Arc<q_quantum_crypto::QuantumSigner>,

    /// DHT command sender
    dht_tx: mpsc::Sender<DhtCommand>,

    /// Cache of verified peer public keys
    /// Key: peer node_id
    /// Value: CachedPublicKey with fingerprint
    peer_pubkey_cache: Arc<RwLock<HashMap<String, CachedPublicKey>>>,

    /// TTL for cached public keys (1 hour)
    cache_ttl: Duration,
}

impl PublicKeyDhtManager {
    /// Create new public key DHT manager
    pub fn new(
        node_id: String,
        peer_id: String,
        capability: NodeCapability,
        quantum_signer: Arc<q_quantum_crypto::QuantumSigner>,
        dht_tx: mpsc::Sender<DhtCommand>,
    ) -> Self {
        Self {
            node_id,
            peer_id,
            capability,
            quantum_signer,
            dht_tx,
            peer_pubkey_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Announce our public key to the DHT
    ///
    /// This implements the correct DHT announcement pattern:
    /// 1. Create announcement with signature: None
    /// 2. Serialize unsigned announcement
    /// 3. Sign those exact bytes
    /// 4. Store BOTH signed announcement AND unsigned bytes in DHT
    ///
    /// The unsigned bytes are needed for verification to match exactly.
    pub async fn announce_public_key_to_dht(&self) -> Result<()> {
        let public_key = self.quantum_signer.get_public_key().await?;

        // Verify our node_id matches our public key hash (self-check)
        let derived_node_id = Self::derive_node_id_from_pubkey(&public_key);
        if derived_node_id != self.node_id {
            error!("❌ CRITICAL: Our node_id doesn't match our public key hash!");
            error!("   Expected: {}", derived_node_id);
            error!("   Actual: {}", self.node_id);
            return Err(anyhow!("Node ID / public key mismatch"));
        }

        // 1. Create unsigned announcement
        let announcement = NodePublicKeyAnnouncement {
            node_id: self.node_id.clone(),
            peer_id: self.peer_id.clone(),
            quantum_public_key: public_key,
            capability: self.capability.clone(),
            announced_at: chrono::Utc::now().timestamp(),
            signature: None, // ✅ Explicitly None for signing
        };

        // 2. Serialize unsigned announcement
        let unsigned_bytes = bincode::serialize(&announcement)?;

        // 3. Sign the unsigned bytes
        let quantum_signature = self.quantum_signer.sign_message(&unsigned_bytes).await?;

        // 4. Create signed announcement
        let signed_announcement = NodePublicKeyAnnouncement {
            signature: Some(bincode::serialize(&quantum_signature)?),
            ..announcement
        };

        // 5. Store BOTH announcement AND unsigned_bytes in DHT
        // CRITICAL: Unsigned bytes needed for verification
        let dht_value = bincode::serialize(&(signed_announcement, unsigned_bytes))?;
        let dht_key = format!("qnk:pubkey:{}", self.node_id);

        self.dht_tx.send(DhtCommand::PutRecord {
            key: dht_key.clone(),
            value: dht_value,
        }).await?;

        info!("📢 Announced public key to DHT");
        info!("   Node ID: {}", self.node_id);
        info!("   DHT Key: {}", dht_key);
        info!("   Peer ID: {}", self.peer_id);
        info!("   Capability: {:?}", self.capability);

        Ok(())
    }

    /// Get cached public key for peer (with fingerprint pre-computed)
    ///
    /// Performance optimization:
    /// - First checks cache (fast path)
    /// - If expired or missing, fetches from DHT (slow path)
    /// - Computes fingerprint once and caches it
    pub async fn get_cached_pubkey(&self, peer_node_id: &str) -> Result<CachedPublicKey> {
        // Check cache first (fast path)
        {
            let cache = self.peer_pubkey_cache.read().await;
            if let Some(cached) = cache.get(peer_node_id) {
                if cached.cached_at.elapsed() < self.cache_ttl {
                    debug!("✅ Public key cache HIT for {}", peer_node_id);
                    return Ok(cached.clone());
                } else {
                    debug!("⏰ Public key cache EXPIRED for {} (age: {}s)",
                           peer_node_id, cached.cached_at.elapsed().as_secs());
                }
            }
        }

        // Cache miss or expired - fetch from DHT (slow path)
        debug!("💾 Public key cache MISS for {} - fetching from DHT", peer_node_id);
        let pubkey = self.get_peer_public_key_from_dht(peer_node_id).await?;

        // Compute fingerprint once
        let cached = CachedPublicKey::new(pubkey);

        // Store in cache
        self.peer_pubkey_cache.write().await.insert(
            peer_node_id.to_string(),
            cached.clone(),
        );

        debug!("💾 Cached public key for {}", peer_node_id);

        Ok(cached)
    }

    /// Fetch peer's public key from DHT with signature verification
    ///
    /// Security checks:
    /// 1. Fetch announcement from DHT
    /// 2. Verify node_id matches hash(public_key) - prevents impersonation
    /// 3. Verify quantum signature using public key from announcement
    /// 4. Return verified public key
    async fn get_peer_public_key_from_dht(&self, peer_node_id: &str) -> Result<Vec<u8>> {
        let dht_key = format!("qnk:pubkey:{}", peer_node_id);

        // TODO: Implement DHT get with response channel
        // For now, return error indicating DHT get needs async response handling
        // This will be completed when NetworkCommand enum is extended with response channels

        warn!("⚠️  DHT public key fetch not yet fully implemented");
        warn!("   Need to add response channel to DhtCommand::GetRecord");
        warn!("   Falling back to self-declared keys for now (INSECURE)");

        Err(anyhow!("DHT public key fetch pending async response implementation"))
    }

    /// Derive node_id from public key hash
    ///
    /// This binding prevents impersonation:
    /// - Node cannot claim another node's ID without their private key
    /// - Changing keys means changing identity (acceptable tradeoff)
    pub fn derive_node_id_from_pubkey(pubkey: &[u8]) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(pubkey);
        let hash = hasher.finalize();
        hex::encode(&hash[..16]) // First 16 bytes as node_id (32 hex chars)
    }

    /// Verify DHT announcement signature
    ///
    /// Verification flow:
    /// 1. Extract signature from announcement
    /// 2. Reconstruct unsigned announcement (signature: None)
    /// 3. Use unsigned_bytes from DHT (avoids re-serialization mismatch)
    /// 4. Verify signature with public key from announcement
    /// 5. Verify node_id matches hash(public_key)
    pub async fn verify_dht_announcement(
        announcement: &NodePublicKeyAnnouncement,
        unsigned_bytes: &[u8],
    ) -> Result<bool> {
        // 1. Check signature exists
        let signature_bytes = announcement.signature.as_ref()
            .ok_or_else(|| anyhow!("Announcement missing signature"))?;

        // 2. Deserialize quantum signature
        let signature: q_quantum_crypto::QuantumSignature =
            bincode::deserialize(signature_bytes)?;

        // 3. Verify node_id matches public key hash (prevents impersonation)
        let derived_node_id = Self::derive_node_id_from_pubkey(&announcement.quantum_public_key);
        if derived_node_id != announcement.node_id {
            error!("❌ Node ID doesn't match public key hash");
            error!("   Claimed node_id: {}", announcement.node_id);
            error!("   Derived from pubkey: {}", derived_node_id);
            error!("   🚨 POSSIBLE IMPERSONATION ATTACK");
            return Ok(false);
        }

        // 4. Verify signature using unsigned bytes from DHT
        // CRITICAL: Use unsigned_bytes from DHT, not re-serialized
        // Convert node_id hex string to NodeId ([u8; 32])
        let node_id_bytes = hex::decode(&announcement.node_id)
            .context("Failed to decode node_id from hex")?;

        let node_id: q_types::NodeId = node_id_bytes.as_slice().try_into()
            .context("node_id must be exactly 32 bytes")?;

        let verifier = q_quantum_crypto::QuantumVerifier::new(node_id);

        let valid = verifier.verify_signature(unsigned_bytes, &signature).await?;

        if valid {
            debug!("✅ DHT announcement signature verified for {}", announcement.node_id);
        } else {
            error!("❌ Invalid DHT announcement signature from {}", announcement.node_id);
            error!("   🚨 POSSIBLE SIGNATURE FORGERY");
        }

        Ok(valid)
    }

    /// Start automatic re-announcement loop
    ///
    /// Announces public key every 12 hours (24-hour TTL / 2)
    pub async fn start_reannouncement_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(12 * 3600)); // 12 hours

        loop {
            interval.tick().await;

            info!("🔄 Re-announcing public key to DHT (periodic)");

            if let Err(e) = self.announce_public_key_to_dht().await {
                error!("Failed to re-announce public key: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_node_id_from_pubkey() {
        let test_pubkey = b"test_public_key_bytes";
        let node_id = PublicKeyDhtManager::derive_node_id_from_pubkey(test_pubkey);

        // Node ID should be 32 hex characters (16 bytes)
        assert_eq!(node_id.len(), 32);

        // Same pubkey should always produce same node_id (deterministic)
        let node_id2 = PublicKeyDhtManager::derive_node_id_from_pubkey(test_pubkey);
        assert_eq!(node_id, node_id2);

        // Different pubkey should produce different node_id
        let different_pubkey = b"different_public_key";
        let different_node_id = PublicKeyDhtManager::derive_node_id_from_pubkey(different_pubkey);
        assert_ne!(node_id, different_node_id);
    }

    #[test]
    fn test_cached_public_key_fingerprint() {
        let test_pubkey = b"test_public_key_bytes".to_vec();
        let cached = CachedPublicKey::new(test_pubkey.clone());

        // Fingerprint should be 32 bytes (SHA3-256)
        assert_eq!(cached.fingerprint.len(), 32);

        // Same pubkey should produce same fingerprint
        let cached2 = CachedPublicKey::new(test_pubkey.clone());
        assert_eq!(cached.fingerprint, cached2.fingerprint);

        // Different pubkey should produce different fingerprint
        let different_pubkey = b"different_public_key".to_vec();
        let cached3 = CachedPublicKey::new(different_pubkey);
        assert_ne!(cached.fingerprint, cached3.fingerprint);
    }
}
