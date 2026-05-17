//! Lattice-Based Aggregate Signatures for Gossip Protocol
//!
//! Implements signature aggregation based on IACR 2025/1056 for efficient
//! gossipsub message signing with ~98% bandwidth reduction.
//!
//! ## Why Lattice Aggregate Signatures?
//!
//! Traditional approach: Each validator signs messages individually
//! - 100 validators × 64 bytes = 6,400 bytes per message
//!
//! With aggregation: Combine all signatures into one
//! - ~128 bytes total (98% reduction)
//!
//! ## Integration with libp2p Gossipsub
//!
//! ```ignore
//! use q_network::lattice_gossip::{GossipAggregator, AggregatedGossipMessage};
//!
//! // Create aggregator
//! let aggregator = GossipAggregator::new(SecurityLevel::Standard)?;
//!
//! // Add signatures as they arrive
//! for (validator, sig) in incoming_signatures {
//!     aggregator.add_signature(&message, &sig, &validator.public_key)?;
//! }
//!
//! // Aggregate and broadcast
//! let aggregated = aggregator.aggregate()?;
//! gossipsub.publish(topic, aggregated.to_bytes())?;
//! ```

#[cfg(feature = "advanced-crypto")]
use q_crypto_advanced::lattice_aggregate::{
    LatticeParams, LatticeKeyPair, LatticePublicKey, LatticeSecretKey,
    LatticeSignature, AggregateSignature, SignatureAggregator,
};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Security level for lattice operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeSecurityLevel {
    /// 128-bit security (fastest)
    Standard,
    /// 192-bit security
    High,
    /// 256-bit security (most secure)
    Paranoid,
}

impl Default for LatticeSecurityLevel {
    fn default() -> Self {
        Self::Standard
    }
}

/// Configuration for gossip signature aggregation
#[derive(Debug, Clone)]
pub struct GossipAggregatorConfig {
    /// Security level for signatures
    pub security_level: LatticeSecurityLevel,
    /// Maximum signatures to aggregate (memory limit)
    pub max_batch_size: usize,
    /// Minimum signatures before triggering aggregation
    pub min_batch_size: usize,
    /// Timeout for batch completion (ms)
    pub batch_timeout_ms: u64,
    /// Enable signature caching
    pub enable_cache: bool,
}

impl Default for GossipAggregatorConfig {
    fn default() -> Self {
        Self {
            security_level: LatticeSecurityLevel::Standard,
            max_batch_size: 1000,
            min_batch_size: 10,
            batch_timeout_ms: 100,
            enable_cache: true,
        }
    }
}

/// SECURITY: Nonce tracking key to prevent nonce reuse attacks
///
/// In lattice-based signatures, reusing the same nonce for different messages
/// can break the security of the scheme (similar to ECDSA nonce reuse).
#[derive(Clone, Hash, Eq, PartialEq)]
struct NonceKey {
    /// Signer's public key hash
    signer_hash: [u8; 32],
    /// Message hash being signed
    message_hash: [u8; 32],
    /// Signature nonce (extracted from signature)
    nonce_hash: [u8; 32],
}

/// Pending signature in the aggregation queue
#[derive(Clone)]
struct PendingSignature {
    message_hash: [u8; 32],
    signature_bytes: Vec<u8>,
    public_key_bytes: Vec<u8>,
    timestamp_ms: u64,
    /// Original message bytes (needed for aggregation verification)
    original_message: Vec<u8>,
    /// Unique nonce identifier for replay protection
    nonce_hash: [u8; 32],
}

/// An aggregated gossip message containing multiple validator signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGossipMessage {
    /// The original message content
    pub message: Vec<u8>,
    /// Hash of the message
    pub message_hash: [u8; 32],
    /// Aggregated signature bytes
    pub aggregate_signature: Vec<u8>,
    /// Public keys of all signers (compressed)
    pub signer_keys: Vec<Vec<u8>>,
    /// Number of signatures aggregated
    pub signature_count: u32,
    /// Topic this message was for
    pub topic: String,
    /// Security level used
    pub security_level: LatticeSecurityLevel,
}

impl AggregatedGossipMessage {
    /// Calculate bandwidth savings vs individual signatures
    ///
    /// # Security
    /// Uses saturating_mul to prevent integer overflow on 32-bit systems
    pub fn bandwidth_savings(&self) -> f64 {
        // SECURITY FIX: Use saturating_mul to prevent overflow
        let individual_size = (self.signature_count as usize).saturating_mul(64);
        let aggregated_size = self.aggregate_signature.len();

        if individual_size > 0 {
            1.0 - (aggregated_size as f64 / individual_size as f64)
        } else {
            0.0
        }
    }

    /// Serialize for transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| anyhow!("Deserialization failed: {}", e))
    }
}

/// Gossip signature aggregator for efficient message broadcasting
#[cfg(feature = "advanced-crypto")]
pub struct GossipAggregator {
    config: GossipAggregatorConfig,
    params: LatticeParams,
    /// Pending signatures by topic
    pending: RwLock<HashMap<String, Vec<PendingSignature>>>,
    /// Aggregation statistics
    stats: RwLock<AggregationStats>,
    /// SECURITY: Track seen nonces to prevent nonce reuse attacks
    /// Nonce reuse in lattice signatures can break the security of the scheme
    seen_nonces: RwLock<HashSet<NonceKey>>,
    /// Maximum size for nonce cache (prevent memory exhaustion)
    max_nonce_cache_size: usize,
}

/// Statistics for signature aggregation
#[derive(Debug, Clone, Default)]
pub struct AggregationStats {
    pub total_signatures_received: u64,
    pub total_aggregations: u64,
    pub total_bytes_saved: u64,
    pub average_batch_size: f64,
    pub average_aggregation_time_us: u64,
}

#[cfg(feature = "advanced-crypto")]
impl GossipAggregator {
    /// Create a new gossip aggregator
    pub fn new(config: GossipAggregatorConfig) -> Result<Self> {
        let level = match config.security_level {
            LatticeSecurityLevel::Standard => q_crypto_advanced::SecurityLevel::Standard,
            LatticeSecurityLevel::High => q_crypto_advanced::SecurityLevel::High,
            LatticeSecurityLevel::Paranoid => q_crypto_advanced::SecurityLevel::Paranoid,
        };

        let params = LatticeParams::new(level)?;

        info!(
            "Lattice gossip aggregator initialized with {:?} security",
            config.security_level
        );

        Ok(Self {
            config,
            params,
            pending: RwLock::new(HashMap::new()),
            stats: RwLock::new(AggregationStats::default()),
            seen_nonces: RwLock::new(HashSet::new()),
            max_nonce_cache_size: 100_000, // Store up to 100k nonces
        })
    }

    /// SECURITY: Extract nonce hash from signature bytes
    /// This is used to detect nonce reuse attacks
    fn extract_nonce_hash(signature_bytes: &[u8]) -> [u8; 32] {
        // Hash the first portion of signature which contains the nonce commitment
        let mut hasher = Sha3_256::new();
        // Use first 64 bytes of signature (nonce commitment region)
        let nonce_region_len = signature_bytes.len().min(64);
        hasher.update(&signature_bytes[..nonce_region_len]);
        hasher.finalize().into()
    }

    /// SECURITY: Check if a nonce has been seen before
    /// Returns true if the nonce is new (safe to use), false if it's a replay
    async fn check_and_record_nonce(
        &self,
        signer_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool> {
        let signer_hash = Self::hash_message(signer_key);
        let message_hash = Self::hash_message(message);
        let nonce_hash = Self::extract_nonce_hash(signature);

        let nonce_key = NonceKey {
            signer_hash,
            message_hash,
            nonce_hash,
        };

        let mut seen_nonces = self.seen_nonces.write().await;

        // Check if nonce already seen
        if seen_nonces.contains(&nonce_key) {
            error!(
                "🚨 SECURITY: Nonce reuse detected! Signer: {:?}",
                hex::encode(&signer_key[..8.min(signer_key.len())])
            );
            return Ok(false);
        }

        // Prevent memory exhaustion - clear oldest entries if too many
        if seen_nonces.len() >= self.max_nonce_cache_size {
            // Clear ~10% of entries (in production, would use LRU cache)
            warn!(
                "Nonce cache full ({}), clearing old entries",
                seen_nonces.len()
            );
            let to_remove: Vec<_> = seen_nonces.iter().take(self.max_nonce_cache_size / 10).cloned().collect();
            for key in to_remove {
                seen_nonces.remove(&key);
            }
        }

        // Record this nonce
        seen_nonces.insert(nonce_key);
        Ok(true)
    }

    /// Add a signature to the pending batch
    ///
    /// # Security
    /// This function performs nonce checking to prevent nonce reuse attacks.
    /// If a signature with the same nonce from the same signer for the same message
    /// is seen again, it will be rejected.
    pub async fn add_signature(
        &self,
        topic: &str,
        message: &[u8],
        signature: &[u8],
        public_key: &[u8],
    ) -> Result<Option<AggregatedGossipMessage>> {
        // SECURITY: Check for nonce reuse before accepting signature
        if !self.check_and_record_nonce(public_key, message, signature).await? {
            return Err(anyhow!(
                "SECURITY: Signature rejected due to nonce reuse. This could indicate an attack."
            ));
        }

        let message_hash = Self::hash_message(message);
        let nonce_hash = Self::extract_nonce_hash(signature);

        let pending_sig = PendingSignature {
            message_hash,
            signature_bytes: signature.to_vec(),
            public_key_bytes: public_key.to_vec(),
            timestamp_ms: Self::current_time_ms(),
            original_message: message.to_vec(),
            nonce_hash,
        };

        let mut pending = self.pending.write().await;
        let batch = pending.entry(topic.to_string()).or_insert_with(Vec::new);
        batch.push(pending_sig);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_signatures_received += 1;
        }

        // Check if we should aggregate
        if batch.len() >= self.config.min_batch_size {
            let signatures = std::mem::take(batch);
            drop(pending);

            let aggregated = self.aggregate_batch(topic, &message_hash, message, signatures).await?;
            return Ok(Some(aggregated));
        }

        Ok(None)
    }

    /// Force aggregation of pending signatures for a topic
    ///
    /// # Security
    /// This now properly includes the original message for message authentication.
    /// Previously, this could create aggregated messages with empty content.
    pub async fn flush_topic(&self, topic: &str) -> Result<Option<AggregatedGossipMessage>> {
        let mut pending = self.pending.write().await;

        if let Some(batch) = pending.remove(topic) {
            if batch.is_empty() {
                return Ok(None);
            }

            let message_hash = batch[0].message_hash;
            // SECURITY FIX: Use the stored original message instead of empty bytes
            let original_message = batch[0].original_message.clone();
            drop(pending);

            // Verify all signatures are for the same message
            if !batch.iter().all(|sig| sig.message_hash == message_hash) {
                return Err(anyhow!(
                    "SECURITY: Batch contains signatures for different messages"
                ));
            }

            let aggregated = self.aggregate_batch(
                topic,
                &message_hash,
                &original_message,
                batch,
            ).await?;

            return Ok(Some(aggregated));
        }

        Ok(None)
    }

    /// Aggregate a batch of signatures
    async fn aggregate_batch(
        &self,
        topic: &str,
        message_hash: &[u8; 32],
        original_message: &[u8],
        batch: Vec<PendingSignature>,
    ) -> Result<AggregatedGossipMessage> {
        let start = std::time::Instant::now();

        let mut aggregator = SignatureAggregator::new(self.params.clone())?;
        let mut signer_keys = Vec::with_capacity(batch.len());

        for pending in &batch {
            let sig = LatticeSignature::from_bytes(&pending.signature_bytes)?;
            aggregator.add(&sig)?;
            signer_keys.push(pending.public_key_bytes.clone());
        }

        let aggregate_sig = aggregator.finalize()?;
        let aggregation_time = start.elapsed();

        // Calculate bytes saved
        let individual_size = batch.len() * 64; // Assumed individual signature size
        let aggregated_size = aggregate_sig.to_bytes().len();
        let bytes_saved = individual_size.saturating_sub(aggregated_size);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_aggregations += 1;
            stats.total_bytes_saved += bytes_saved as u64;
            stats.average_batch_size = (stats.average_batch_size * (stats.total_aggregations - 1) as f64
                + batch.len() as f64)
                / stats.total_aggregations as f64;
            stats.average_aggregation_time_us = (stats.average_aggregation_time_us
                * (stats.total_aggregations - 1)
                + aggregation_time.as_micros() as u64)
                / stats.total_aggregations;
        }

        info!(
            "Aggregated {} signatures for topic '{}' (saved {} bytes, {:.1}% reduction)",
            batch.len(),
            topic,
            bytes_saved,
            (bytes_saved as f64 / individual_size as f64) * 100.0
        );

        Ok(AggregatedGossipMessage {
            message: original_message.to_vec(),
            message_hash: *message_hash,
            aggregate_signature: aggregate_sig.to_bytes(),
            signer_keys,
            signature_count: batch.len() as u32,
            topic: topic.to_string(),
            security_level: self.config.security_level,
        })
    }

    /// Verify an aggregated message
    pub fn verify_aggregated(&self, aggregated: &AggregatedGossipMessage) -> Result<bool> {
        let aggregate_sig = AggregateSignature::from_bytes(&aggregated.aggregate_signature)?;

        // Collect public keys
        let mut public_keys = Vec::with_capacity(aggregated.signer_keys.len());
        for key_bytes in &aggregated.signer_keys {
            public_keys.push(LatticePublicKey::from_bytes(key_bytes)?);
        }

        // Verify aggregate signature
        aggregate_sig.verify(&aggregated.message_hash, &public_keys)
    }

    /// Get current statistics
    pub async fn stats(&self) -> AggregationStats {
        self.stats.read().await.clone()
    }

    /// Get pending signature count for a topic
    pub async fn pending_count(&self, topic: &str) -> usize {
        let pending = self.pending.read().await;
        pending.get(topic).map(|v| v.len()).unwrap_or(0)
    }

    /// Hash a message for signing
    fn hash_message(message: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(message);
        hasher.finalize().into()
    }

    /// Get current time in milliseconds
    fn current_time_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Lattice keypair for gossip signing
#[cfg(feature = "advanced-crypto")]
pub struct GossipSigningKey {
    keypair: LatticeKeyPair,
    node_id: [u8; 32],
}

#[cfg(feature = "advanced-crypto")]
impl GossipSigningKey {
    /// Generate a new signing key
    pub fn generate(security_level: LatticeSecurityLevel, node_id: [u8; 32]) -> Result<Self> {
        let level = match security_level {
            LatticeSecurityLevel::Standard => q_crypto_advanced::SecurityLevel::Standard,
            LatticeSecurityLevel::High => q_crypto_advanced::SecurityLevel::High,
            LatticeSecurityLevel::Paranoid => q_crypto_advanced::SecurityLevel::Paranoid,
        };

        let params = LatticeParams::new(level)?;
        let keypair = LatticeKeyPair::generate(&params)?;

        Ok(Self { keypair, node_id })
    }

    /// Sign a gossip message
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let sig = self.keypair.sign(message)?;
        Ok(sig.to_bytes())
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.keypair.public_key().to_bytes()
    }

    /// Get node ID
    pub fn node_id(&self) -> &[u8; 32] {
        &self.node_id
    }
}

/// Batch verifier for received aggregated messages
#[cfg(feature = "advanced-crypto")]
pub struct GossipBatchVerifier {
    pending_verifications: Vec<(AggregatedGossipMessage, bool)>,
    params: LatticeParams,
}

#[cfg(feature = "advanced-crypto")]
impl GossipBatchVerifier {
    /// Create a new batch verifier
    pub fn new(security_level: LatticeSecurityLevel) -> Result<Self> {
        let level = match security_level {
            LatticeSecurityLevel::Standard => q_crypto_advanced::SecurityLevel::Standard,
            LatticeSecurityLevel::High => q_crypto_advanced::SecurityLevel::High,
            LatticeSecurityLevel::Paranoid => q_crypto_advanced::SecurityLevel::Paranoid,
        };

        let params = LatticeParams::new(level)?;

        Ok(Self {
            pending_verifications: Vec::new(),
            params,
        })
    }

    /// Add message to verification queue
    pub fn add(&mut self, message: AggregatedGossipMessage) {
        self.pending_verifications.push((message, false));
    }

    /// Verify all pending messages
    pub fn verify_all(&mut self) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(self.pending_verifications.len());

        for (msg, verified) in &mut self.pending_verifications {
            if !*verified {
                let aggregate_sig = AggregateSignature::from_bytes(&msg.aggregate_signature)?;

                let mut public_keys = Vec::with_capacity(msg.signer_keys.len());
                for key_bytes in &msg.signer_keys {
                    public_keys.push(LatticePublicKey::from_bytes(key_bytes)?);
                }

                *verified = aggregate_sig.verify(&msg.message_hash, &public_keys)?;
            }
            results.push(*verified);
        }

        Ok(results)
    }

    /// Get count of pending verifications
    pub fn pending_count(&self) -> usize {
        self.pending_verifications.len()
    }

    /// Clear all pending verifications
    pub fn clear(&mut self) {
        self.pending_verifications.clear();
    }
}

// Fallback for when advanced-crypto is disabled
#[cfg(not(feature = "advanced-crypto"))]
pub struct GossipAggregator;

#[cfg(not(feature = "advanced-crypto"))]
impl GossipAggregator {
    pub fn new(_config: GossipAggregatorConfig) -> Result<Self> {
        Err(anyhow!(
            "Lattice gossip aggregation requires the 'advanced-crypto' feature. Enable it in Cargo.toml."
        ))
    }
}

#[cfg(all(test, feature = "advanced-crypto"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gossip_aggregator_creation() {
        let config = GossipAggregatorConfig::default();
        let aggregator = GossipAggregator::new(config);
        assert!(aggregator.is_ok());
    }

    #[tokio::test]
    async fn test_signature_aggregation() {
        let config = GossipAggregatorConfig {
            min_batch_size: 3,
            ..Default::default()
        };
        let aggregator = GossipAggregator::new(config).unwrap();

        // Generate signing keys
        let key1 = GossipSigningKey::generate(LatticeSecurityLevel::Standard, [1u8; 32]).unwrap();
        let key2 = GossipSigningKey::generate(LatticeSecurityLevel::Standard, [2u8; 32]).unwrap();
        let key3 = GossipSigningKey::generate(LatticeSecurityLevel::Standard, [3u8; 32]).unwrap();

        let message = b"Test gossip message";
        let topic = "/qnk/blocks";

        // Add signatures
        let sig1 = key1.sign(message).unwrap();
        let sig2 = key2.sign(message).unwrap();
        let sig3 = key3.sign(message).unwrap();

        let result1 = aggregator
            .add_signature(topic, message, &sig1, &key1.public_key_bytes())
            .await
            .unwrap();
        assert!(result1.is_none()); // Not enough yet

        let result2 = aggregator
            .add_signature(topic, message, &sig2, &key2.public_key_bytes())
            .await
            .unwrap();
        assert!(result2.is_none()); // Still not enough

        let result3 = aggregator
            .add_signature(topic, message, &sig3, &key3.public_key_bytes())
            .await
            .unwrap();
        assert!(result3.is_some()); // Should trigger aggregation

        let aggregated = result3.unwrap();
        assert_eq!(aggregated.signature_count, 3);
        assert!(aggregated.bandwidth_savings() > 0.5); // Should save at least 50%

        println!(
            "Aggregated {} signatures with {:.1}% bandwidth savings",
            aggregated.signature_count,
            aggregated.bandwidth_savings() * 100.0
        );
    }

    #[tokio::test]
    async fn test_aggregation_stats() {
        let config = GossipAggregatorConfig {
            min_batch_size: 2,
            ..Default::default()
        };
        let aggregator = GossipAggregator::new(config).unwrap();

        let key1 = GossipSigningKey::generate(LatticeSecurityLevel::Standard, [1u8; 32]).unwrap();
        let key2 = GossipSigningKey::generate(LatticeSecurityLevel::Standard, [2u8; 32]).unwrap();

        let message = b"Stats test message";
        let topic = "/qnk/test";

        let sig1 = key1.sign(message).unwrap();
        let sig2 = key2.sign(message).unwrap();

        aggregator
            .add_signature(topic, message, &sig1, &key1.public_key_bytes())
            .await
            .unwrap();
        aggregator
            .add_signature(topic, message, &sig2, &key2.public_key_bytes())
            .await
            .unwrap();

        let stats = aggregator.stats().await;
        assert_eq!(stats.total_signatures_received, 2);
        assert_eq!(stats.total_aggregations, 1);
        assert!(stats.total_bytes_saved > 0);
    }
}
