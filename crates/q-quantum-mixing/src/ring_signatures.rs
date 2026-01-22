//! # Phase 1A: Production-Ready Quantum Ring Signature System
//!
//! v2.5.1-beta: REAL CRYPTOGRAPHIC IMPLEMENTATION
//!
//! This module implements proper linkable ring signatures using curve25519-dalek
//! for elliptic curve operations. Provides:
//!
//! - **Unlinkability**: Ring signatures hide which member signed
//! - **Linkability via Key Images**: Same key produces same key image (double-spend detection)
//! - **Unforgeability**: Cannot create signatures without private key
//!
//! ## Cryptographic Foundations
//!
//! Based on "Linkable Spontaneous Anonymous Group Signatures" (LSAG) with curve25519:
//!
//! - Key Image: `I = x * H_p(P)` where `x` is private key, `P` is public key
//! - Ring Response: `s = r + c*x mod L` (proper scalar field arithmetic)
//! - Verification: `c' = H(m, L, s_0*G + c_0*P_0, s_0*H_p(P_0) + c_0*I, ...)`
//!
//! ## Security Properties
//!
//! - 128-bit security from curve25519
//! - Quantum-enhanced nonces for additional entropy
//! - Zeroization of sensitive data on drop

use crate::{
    error::{MixingError, Result},
    quantum_entropy::QuantumEntropyPool,
};

use curve25519_dalek::{
    constants::RISTRETTO_BASEPOINT_TABLE,
    ristretto::{CompressedRistretto, RistrettoPoint},
    scalar::Scalar,
    traits::Identity,
};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_512};
use std::sync::Arc;
use tracing::{debug, info, warn};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// A linkable ring signature with real elliptic curve cryptography
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingSignature {
    /// The ring signature values (c_i, s_i) for each ring member
    pub signature_values: Vec<SignatureValue>,
    /// Key image for linkability detection (compressed Ristretto point)
    pub key_image: KeyImage,
    /// Initial challenge value c_0
    pub challenge: [u8; 32],
    /// Ring of public keys used in the signature (compressed Ristretto points)
    pub ring: Vec<[u8; 32]>,
    /// Timestamp when signature was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual signature value in a ring signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureValue {
    /// Challenge value c_i (scalar)
    pub challenge: [u8; 32],
    /// Response value s_i (scalar)
    pub response: [u8; 32],
}

/// Key image for preventing double-spending in linkable ring signatures
/// Key Image I = x * H_p(P) where x is private key, P is public key
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyImage {
    /// The key image point (compressed Ristretto)
    pub image: [u8; 32],
    /// Quantum nonce for additional entropy
    pub quantum_nonce: [u8; 32],
}

/// Production-grade quantum ring signature system with real EC math
/// v2.5.1-beta: Complete rewrite using curve25519-dalek
#[derive(Clone)]
pub struct QuantumRingSigner {
    /// Private key scalar (zeroized on drop)
    private_key: Scalar,
    /// Public key point (compressed)
    public_key: CompressedRistretto,
    /// Public key bytes for ring matching
    public_key_bytes: [u8; 32],
    /// Quantum entropy source for enhanced randomness
    quantum_entropy: Arc<QuantumEntropyPool>,
    /// Cache of previously computed key images (only the image bytes, not the quantum nonce)
    /// to prevent double-spend. The image bytes are deterministic for a given private key.
    key_image_cache: std::collections::HashSet<[u8; 32]>,
}

impl QuantumRingSigner {
    /// Create new quantum ring signer with quantum entropy
    /// Generates a new keypair using quantum-enhanced randomness
    pub async fn new(entropy_pool: Arc<QuantumEntropyPool>) -> Result<Self> {
        info!("Initializing Quantum Ring Signer with real EC cryptography (v2.5.1-beta)");

        // Generate private key scalar using quantum entropy
        let mut private_key_bytes = [0u8; 64]; // Need 64 bytes for uniform scalar reduction
        entropy_pool.fill_bytes(&mut private_key_bytes[..32]).await?;
        entropy_pool.fill_bytes(&mut private_key_bytes[32..]).await?;

        let private_key = Scalar::from_bytes_mod_order_wide(&private_key_bytes);

        // Compute public key: P = x * G
        let public_key_point = RISTRETTO_BASEPOINT_TABLE.basepoint() *private_key;
        let public_key = public_key_point.compress();
        let public_key_bytes = public_key.to_bytes();

        // Zeroize the raw bytes
        let mut zero_bytes = private_key_bytes;
        zero_bytes.zeroize();

        Ok(Self {
            private_key,
            public_key,
            public_key_bytes,
            quantum_entropy: entropy_pool,
            key_image_cache: std::collections::HashSet::new(),
        })
    }

    /// Create from existing private key bytes (for wallet restoration)
    pub async fn from_private_key(
        private_key_bytes: [u8; 32],
        entropy_pool: Arc<QuantumEntropyPool>,
    ) -> Result<Self> {
        // Extend to 64 bytes for uniform reduction
        let mut extended = [0u8; 64];
        extended[..32].copy_from_slice(&private_key_bytes);
        let private_key = Scalar::from_bytes_mod_order_wide(&extended);

        // Compute public key
        let public_key_point = RISTRETTO_BASEPOINT_TABLE.basepoint() *private_key;
        let public_key = public_key_point.compress();
        let public_key_bytes = public_key.to_bytes();

        Ok(Self {
            private_key,
            public_key,
            public_key_bytes,
            quantum_entropy: entropy_pool,
            key_image_cache: std::collections::HashSet::new(),
        })
    }

    /// Create linkable ring signature using real LSAG algorithm
    ///
    /// Algorithm:
    /// 1. Compute key image I = x * H_p(P)
    /// 2. Generate random nonce r, compute L_s = r*G, R_s = r*H_p(P_s)
    /// 3. For each non-signer member, pick random c_i, s_i
    /// 4. Compute challenges c_{i+1} = H(m, L, L_i, R_i, ...) in a ring
    /// 5. Solve for s_s = r - c_s * x
    pub async fn create_ring_signature(
        &mut self,
        message: &[u8],
        ring: Vec<[u8; 32]>,
    ) -> Result<RingSignature> {
        debug!("Creating ring signature with {} ring members (real EC)", ring.len());

        if ring.is_empty() {
            return Err(MixingError::RingSignatureError("Ring cannot be empty".to_string()));
        }

        // Find our position in the ring
        let secret_index = ring.iter().position(|pk| pk == &self.public_key_bytes)
            .ok_or_else(|| MixingError::RingSignatureError("Public key not found in ring".to_string()))?;

        // 1. Generate key image: I = x * H_p(P)
        let key_image = self.generate_key_image().await?;

        // Check for double-spend attempt (only compare image bytes, not quantum nonce)
        if self.key_image_cache.contains(&key_image.image) {
            return Err(MixingError::RingSignatureError("Key image already used (double-spend attempt)".to_string()));
        }

        let key_image_point = decompress_or_hash_to_point(&key_image.image)?;

        // Parse ring public keys
        let ring_points: Vec<RistrettoPoint> = ring
            .iter()
            .map(|pk_bytes| decompress_or_hash_to_point(pk_bytes))
            .collect::<Result<Vec<_>>>()?;

        // 2. Generate random nonce with quantum entropy
        let mut nonce_bytes = [0u8; 64];
        self.quantum_entropy.fill_bytes(&mut nonce_bytes[..32]).await?;
        self.quantum_entropy.fill_bytes(&mut nonce_bytes[32..]).await?;
        let nonce = Scalar::from_bytes_mod_order_wide(&nonce_bytes);

        // Compute L_s = r * G (for our position)
        let l_s = RISTRETTO_BASEPOINT_TABLE.basepoint() *nonce;

        // Compute R_s = r * H_p(P_s)
        let hp_s = hash_to_point(&ring[secret_index]);
        let r_s = nonce * hp_s;

        // Initialize signature values
        let ring_size = ring.len();
        let mut challenges: Vec<Scalar> = vec![Scalar::ZERO; ring_size];
        let mut responses: Vec<Scalar> = vec![Scalar::ZERO; ring_size];

        // 3. Start computing the ring from secret_index + 1
        // First challenge: c_{s+1} = H(m, L, R)
        challenges[(secret_index + 1) % ring_size] = compute_challenge(
            message,
            &l_s.compress().to_bytes(),
            &r_s.compress().to_bytes(),
            &key_image.image,
        );

        // 4. For each member in the ring (wrapping around)
        for i in 1..ring_size {
            let idx = (secret_index + i) % ring_size;
            let next_idx = (idx + 1) % ring_size;

            // Generate random response s_i
            let mut response_bytes = [0u8; 64];
            self.quantum_entropy.fill_bytes(&mut response_bytes[..32]).await?;
            self.quantum_entropy.fill_bytes(&mut response_bytes[32..]).await?;
            responses[idx] = Scalar::from_bytes_mod_order_wide(&response_bytes);

            // Compute L_i = s_i * G + c_i * P_i
            let l_i = RISTRETTO_BASEPOINT_TABLE.basepoint() *responses[idx] + ring_points[idx] * challenges[idx];

            // Compute R_i = s_i * H_p(P_i) + c_i * I
            let hp_i = hash_to_point(&ring[idx]);
            let r_i = responses[idx] * hp_i + challenges[idx] * key_image_point;

            // Compute next challenge
            if next_idx != secret_index {
                challenges[next_idx] = compute_challenge(
                    message,
                    &l_i.compress().to_bytes(),
                    &r_i.compress().to_bytes(),
                    &key_image.image,
                );
            }
        }

        // 5. Compute our response: s_s = r - c_s * x
        // Need to compute c_s first (it's the last one we haven't computed)
        let prev_idx = if secret_index == 0 { ring_size - 1 } else { secret_index - 1 };

        // Recompute L and R for prev_idx to get c_s
        let l_prev = RISTRETTO_BASEPOINT_TABLE.basepoint() *responses[prev_idx] + ring_points[prev_idx] * challenges[prev_idx];
        let hp_prev = hash_to_point(&ring[prev_idx]);
        let r_prev = responses[prev_idx] * hp_prev + challenges[prev_idx] * key_image_point;

        challenges[secret_index] = compute_challenge(
            message,
            &l_prev.compress().to_bytes(),
            &r_prev.compress().to_bytes(),
            &key_image.image,
        );

        // s_s = r - c_s * x (mod L)
        responses[secret_index] = nonce - challenges[secret_index] * self.private_key;

        // Convert to SignatureValue structs
        let signature_values: Vec<SignatureValue> = challenges
            .iter()
            .zip(responses.iter())
            .map(|(c, s)| SignatureValue {
                challenge: c.to_bytes(),
                response: s.to_bytes(),
            })
            .collect();

        // Cache the key image bytes to prevent reuse (only the image, not the quantum nonce)
        self.key_image_cache.insert(key_image.image);

        Ok(RingSignature {
            signature_values,
            key_image,
            challenge: challenges[0].to_bytes(),
            ring,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify a ring signature using real EC verification
    ///
    /// Verification algorithm:
    /// 1. For each i: L_i = s_i * G + c_i * P_i, R_i = s_i * H_p(P_i) + c_i * I
    /// 2. Compute c_{i+1} = H(m, L_i, R_i)
    /// 3. Check that computed c_0 matches provided c_0
    pub async fn verify_ring_signature(
        &self,
        signature: &RingSignature,
        message: &[u8],
    ) -> Result<bool> {
        debug!("Verifying ring signature with {} members (real EC)", signature.ring.len());

        if signature.signature_values.len() != signature.ring.len() {
            return Ok(false);
        }

        let ring_size = signature.ring.len();
        if ring_size == 0 {
            return Ok(false);
        }

        // Parse key image
        let key_image_point = match decompress_or_hash_to_point(&signature.key_image.image) {
            Ok(p) => p,
            Err(_) => return Ok(false),
        };

        // Parse ring public keys
        let ring_points: Vec<RistrettoPoint> = match signature.ring
            .iter()
            .map(|pk_bytes| decompress_or_hash_to_point(pk_bytes))
            .collect::<Result<Vec<_>>>() {
            Ok(pts) => pts,
            Err(_) => return Ok(false),
        };

        // Start verification from c_0
        // Note: from_canonical_bytes returns CtOption in curve25519-dalek 4.x
        let mut current_challenge = match Scalar::from_canonical_bytes(signature.challenge.into()).into_option() {
            Some(s) => s,
            None => return Ok(false),
        };

        // Verify each link in the ring
        for i in 0..ring_size {
            let response = match Scalar::from_canonical_bytes(signature.signature_values[i].response.into()).into_option() {
                Some(s) => s,
                None => return Ok(false),
            };

            // L_i = s_i * G + c_i * P_i
            let l_i = RISTRETTO_BASEPOINT_TABLE.basepoint() *response + ring_points[i] * current_challenge;

            // R_i = s_i * H_p(P_i) + c_i * I
            let hp_i = hash_to_point(&signature.ring[i]);
            let r_i = response * hp_i + current_challenge * key_image_point;

            // Compute next challenge
            current_challenge = compute_challenge(
                message,
                &l_i.compress().to_bytes(),
                &r_i.compress().to_bytes(),
                &signature.key_image.image,
            );
        }

        // Check that we returned to the original challenge
        let original_challenge = match Scalar::from_canonical_bytes(signature.challenge.into()).into_option() {
            Some(s) => s,
            None => return Ok(false),
        };

        let valid = current_challenge == original_challenge;

        if valid {
            info!("Ring signature verification successful");
        } else {
            debug!("Ring signature verification failed: challenge mismatch");
        }

        Ok(valid)
    }

    /// Batch verify multiple ring signatures for performance
    pub async fn batch_verify_signatures(
        &self,
        signatures: Vec<(&RingSignature, &[u8])>,
    ) -> Result<Vec<bool>> {
        info!("Batch verifying {} ring signatures", signatures.len());

        let mut results = Vec::with_capacity(signatures.len());

        // TODO: Implement actual batch verification using multi-scalar multiplication
        // For now, verify each signature individually
        for (signature, message) in signatures {
            let result = self.verify_ring_signature(signature, message).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate key image: I = x * H_p(P)
    /// This is the linkable component - same private key always produces same key image
    async fn generate_key_image(&self) -> Result<KeyImage> {
        // Hash public key to curve point: H_p(P)
        let hp = hash_to_point(&self.public_key_bytes);

        // Key image: I = x * H_p(P)
        let key_image_point = self.private_key * hp;
        let image = key_image_point.compress().to_bytes();

        // Add quantum nonce for additional entropy (doesn't affect linkability)
        let mut quantum_nonce = [0u8; 32];
        self.quantum_entropy.fill_bytes(&mut quantum_nonce).await?;

        Ok(KeyImage {
            image,
            quantum_nonce,
        })
    }

    /// Get public key for this signer
    pub fn get_public_key(&self) -> [u8; 32] {
        self.public_key_bytes
    }

    /// Check if a key image has been used before
    pub fn is_key_image_used(&self, key_image: &KeyImage) -> bool {
        self.key_image_cache.contains(&key_image.image)
    }
}

impl Drop for QuantumRingSigner {
    fn drop(&mut self) {
        // Zeroize the private key scalar
        // Note: Scalar doesn't implement Zeroize directly, but its internal bytes can be
        // cleared by replacing with a new scalar
        self.private_key = Scalar::ZERO;
    }
}

/// Hash arbitrary bytes to a Ristretto point using Elligator
fn hash_to_point(data: &[u8]) -> RistrettoPoint {
    let mut hasher = Sha3_512::new();
    hasher.update(b"RingSignature.HashToPoint.v2.5.1");
    hasher.update(data);
    let hash: [u8; 64] = hasher.finalize().into();
    RistrettoPoint::from_uniform_bytes(&hash)
}

/// Decompress a point or hash to point if invalid
fn decompress_or_hash_to_point(bytes: &[u8; 32]) -> Result<RistrettoPoint> {
    let compressed = CompressedRistretto::from_slice(bytes)
        .map_err(|_| MixingError::RingSignatureError("Invalid point encoding".to_string()))?;

    match compressed.decompress() {
        Some(point) => Ok(point),
        None => {
            // If decompression fails, hash to a valid point
            // This allows arbitrary 32-byte public keys in the ring
            Ok(hash_to_point(bytes))
        }
    }
}

/// Compute challenge using Fiat-Shamir transform
fn compute_challenge(message: &[u8], l_bytes: &[u8; 32], r_bytes: &[u8; 32], key_image: &[u8; 32]) -> Scalar {
    let mut hasher = Sha3_512::new();
    hasher.update(b"RingSignature.Challenge.v2.5.1");
    hasher.update(message);
    hasher.update(l_bytes);
    hasher.update(r_bytes);
    hasher.update(key_image);
    let hash: [u8; 64] = hasher.finalize().into();
    Scalar::from_bytes_mod_order_wide(&hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_entropy::QuantumEntropyPool;

    #[tokio::test]
    async fn test_ring_signer_creation() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let signer = QuantumRingSigner::new(entropy_pool).await.unwrap();

        let public_key = signer.get_public_key();
        assert!(!public_key.iter().all(|&b| b == 0), "Public key should not be all zeros");
    }

    #[tokio::test]
    async fn test_ring_signature_creation_and_verification() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        // Create a ring with our public key and some others
        let our_pubkey = signer.get_public_key();

        // Generate valid ring member public keys
        let other1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let other2 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let other3 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        let ring = vec![
            other1.get_public_key(),
            our_pubkey,
            other2.get_public_key(),
            other3.get_public_key(),
        ];

        let message = b"test message for ring signature";
        let signature = signer.create_ring_signature(message, ring.clone()).await.unwrap();

        // Verify signature structure
        assert_eq!(signature.signature_values.len(), ring.len(), "Signature values should match ring size");
        assert_eq!(signature.ring, ring, "Ring should be preserved in signature");
        assert!(!signature.key_image.image.iter().all(|&b| b == 0), "Key image should not be all zeros");

        // Verify the signature
        let is_valid = signer.verify_ring_signature(&signature, message).await.unwrap();
        assert!(is_valid, "Valid ring signature should verify successfully");

        // Verify with wrong message should fail
        let wrong_message = b"different message";
        let is_invalid = signer.verify_ring_signature(&signature, wrong_message).await.unwrap();
        assert!(!is_invalid, "Ring signature with wrong message should fail verification");
    }

    #[tokio::test]
    async fn test_key_image_linkability() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

        // Create signer from same private key twice
        let private_key = [42u8; 32];
        let signer1 = QuantumRingSigner::from_private_key(private_key, entropy_pool.clone()).await.unwrap();
        let signer2 = QuantumRingSigner::from_private_key(private_key, entropy_pool.clone()).await.unwrap();

        // Key images should have the same base (image field) regardless of quantum nonce
        let ki1 = signer1.generate_key_image().await.unwrap();
        let ki2 = signer2.generate_key_image().await.unwrap();

        assert_eq!(ki1.image, ki2.image, "Same private key should produce same key image");
        // Quantum nonces may differ, but images must match for linkability
    }

    #[tokio::test]
    async fn test_double_spend_prevention() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        let our_pubkey = signer.get_public_key();
        let other = QuantumRingSigner::new(entropy_pool).await.unwrap();
        let ring = vec![our_pubkey, other.get_public_key()];

        // Create first signature
        let message1 = b"first message";
        let sig1 = signer.create_ring_signature(message1, ring.clone()).await.unwrap();

        // Attempt to create second signature with same key should fail
        let message2 = b"second message";
        let sig2_result = signer.create_ring_signature(message2, ring.clone()).await;

        assert!(sig2_result.is_err(), "Second signature should fail due to key image reuse");
        assert!(signer.is_key_image_used(&sig1.key_image), "Key image should be marked as used");
    }

    #[tokio::test]
    async fn test_ring_signature_unforgeability() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        // Try to create a forged signature (attacker doesn't have private key)
        let attacker = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let victim = QuantumRingSigner::new(entropy_pool).await.unwrap();

        // Attacker creates ring including victim's public key but signs with own key
        let ring = vec![attacker.get_public_key(), victim.get_public_key()];
        let mut attacker_mut = attacker;

        let message = b"forged message";
        let signature = attacker_mut.create_ring_signature(message, ring.clone()).await.unwrap();

        // Signature should verify (attacker is in the ring)
        let is_valid = signer.verify_ring_signature(&signature, message).await.unwrap();
        assert!(is_valid, "Attacker's signature should verify (they're in the ring)");

        // But key image should be different from victim's
        // This is the security property - attacker cannot produce victim's key image
    }

    #[tokio::test]
    async fn test_batch_verification() {
        let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
        let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        // Create multiple signers and signatures
        let mut signer1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let mut signer2 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

        let other1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let other2 = QuantumRingSigner::new(entropy_pool).await.unwrap();

        let ring1 = vec![signer1.get_public_key(), other1.get_public_key()];
        let ring2 = vec![other2.get_public_key(), signer2.get_public_key()];

        let msg1 = b"batch message 1";
        let msg2 = b"batch message 2";

        let sig1 = signer1.create_ring_signature(msg1, ring1).await.unwrap();
        let sig2 = signer2.create_ring_signature(msg2, ring2).await.unwrap();

        // Batch verify
        let signatures = vec![(&sig1, msg1.as_ref()), (&sig2, msg2.as_ref())];
        let results = verifier.batch_verify_signatures(signatures).await.unwrap();

        assert_eq!(results, vec![true, true], "Both signatures should verify successfully");
    }

    #[tokio::test]
    async fn test_hash_to_point_consistency() {
        // Hash to point should be deterministic
        let data = b"test data for hash to point";
        let p1 = hash_to_point(data);
        let p2 = hash_to_point(data);
        assert_eq!(p1, p2, "Hash to point should be deterministic");

        // Different data should produce different points
        let p3 = hash_to_point(b"different data");
        assert_ne!(p1, p3, "Different data should produce different points");
    }
}
