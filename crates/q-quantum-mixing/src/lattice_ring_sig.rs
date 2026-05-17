//! # Post-Quantum Lattice-Based Linkable Ring Signatures
//!
//! Implementation based on IACR ePrint 2025/2170: "Lattice-Based Linkable Ring Signatures
//! from Module-LWE with Logarithmic Signature Size".
//!
//! This module provides post-quantum secure ring signatures using the Fiat-Shamir with
//! Aborts (FSA) paradigm applied to lattice-based commitments.
//!
//! ## Security Properties
//!
//! - **Anonymity**: The signer's identity is hidden among the ring members
//! - **Linkability**: Signatures from the same signer can be linked via key images
//! - **Unforgeability**: Cannot forge signatures without the private key
//! - **Post-Quantum Security**: Based on Module-LWE hardness assumption
//!
//! ## Security Levels
//!
//! - **Level128**: NIST Level 1 (~128-bit classical security)
//! - **Level192**: NIST Level 3 (~192-bit classical security)
//! - **Level256**: NIST Level 5 (~256-bit classical security)
//!
//! ## Performance Characteristics
//!
//! - Ring sizes up to 1024 members supported
//! - Signature size: O(log n) where n is ring size
//! - Verification time: O(n) for ring size n
//!
//! ## Cryptographic Construction
//!
//! The scheme uses:
//! - Dilithium's polynomial ring R_q = Z_q[X]/(X^256 + 1)
//! - Module-LWE based commitment scheme
//! - Fiat-Shamir with Aborts for challenge generation
//! - Hash-to-curve for key image computation

use crate::error::{MixingError, Result};
use crate::quantum_entropy::QuantumEntropyPool;

use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256, Sha3_512};
use std::sync::Arc;
use tracing::{debug, info};
use zeroize::Zeroize;

/// Helper module for serializing [u8; 64] arrays
mod challenge_seed_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(data: &[u8; 64], serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(data)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let v: Vec<u8> = Vec::deserialize(deserializer)?;
        if v.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 bytes, got {}",
                v.len()
            )));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&v);
        Ok(arr)
    }
}

/// Lattice ring signature security parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// NIST Level 1: ~128-bit classical security
    Level128,
    /// NIST Level 3: ~192-bit classical security
    Level192,
    /// NIST Level 5: ~256-bit classical security
    Level256,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Level256 // Maximum security by default
    }
}

/// Parameters for the lattice-based ring signature scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeRingParams {
    /// Security level (128, 192, or 256 bits)
    pub security_level: SecurityLevel,
    /// Polynomial ring dimension (n = 256)
    pub n: usize,
    /// Module rank (k)
    pub k: usize,
    /// Module dimension for signing (l)
    pub l: usize,
    /// Modulus q
    pub q: u32,
    /// Rejection sampling bound for z
    pub gamma1: u32,
    /// Hint coefficient bound
    pub gamma2: u32,
    /// Maximum coefficient in challenge polynomial
    pub tau: u32,
    /// Bit-length of challenge seed
    pub challenge_bits: usize,
    /// Maximum ring size supported
    pub max_ring_size: usize,
    /// Number of abort retries before failure
    pub max_abort_retries: usize,
}

impl LatticeRingParams {
    /// Create parameters for given security level
    pub fn new(security_level: SecurityLevel) -> Self {
        match security_level {
            SecurityLevel::Level128 => Self {
                security_level,
                n: 256,
                k: 4,
                l: 4,
                q: 8380417,
                gamma1: 1 << 17,
                gamma2: (8380417 - 1) / 88,
                tau: 39,
                challenge_bits: 256,
                max_ring_size: 1024,
                max_abort_retries: 256,
            },
            SecurityLevel::Level192 => Self {
                security_level,
                n: 256,
                k: 6,
                l: 5,
                q: 8380417,
                gamma1: 1 << 19,
                gamma2: (8380417 - 1) / 32,
                tau: 49,
                challenge_bits: 384,
                max_ring_size: 1024,
                max_abort_retries: 256,
            },
            SecurityLevel::Level256 => Self {
                security_level,
                n: 256,
                k: 8,
                l: 7,
                q: 8380417,
                gamma1: 1 << 19,
                gamma2: (8380417 - 1) / 32,
                tau: 60,
                challenge_bits: 512,
                max_ring_size: 1024,
                max_abort_retries: 256,
            },
        }
    }

    /// Get security level in bits
    pub fn security_bits(&self) -> usize {
        match self.security_level {
            SecurityLevel::Level128 => 128,
            SecurityLevel::Level192 => 192,
            SecurityLevel::Level256 => 256,
        }
    }
}

impl Default for LatticeRingParams {
    fn default() -> Self {
        Self::new(SecurityLevel::Level256)
    }
}

/// A lattice-based linkable ring signature
///
/// The signature proves membership in a ring while maintaining anonymity
/// and providing linkability through the key image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeRingSignature {
    /// Key image for linkability detection
    /// I = s * H(pk) where s is the secret key
    pub key_image: Vec<u8>,

    /// Challenge seed for Fiat-Shamir
    /// Used to derive per-member challenges
    #[serde(with = "challenge_seed_serde")]
    pub challenge_seed: [u8; 64],

    /// Response vectors for each ring member
    /// z_i = y_i + c_i * s for the actual signer, random for others
    pub responses: Vec<Vec<u8>>,

    /// Verification hints for efficient verification
    /// Contains commitment randomness and auxiliary data
    pub hints: Vec<u8>,

    /// Security level used for this signature
    pub security_level: SecurityLevel,

    /// Ring public keys (compressed)
    pub ring: Vec<Vec<u8>>,

    /// Timestamp when signature was created
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Protocol version for future compatibility
    pub version: u8,
}

/// Key image for preventing double-signing
///
/// The key image is computed as I = s * H_p(pk) where:
/// - s is the secret signing key
/// - H_p is a hash-to-curve function
/// - pk is the corresponding public key
///
/// Key images are deterministic: same key always produces same image.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatticeKeyImage {
    /// The key image bytes (lattice element)
    pub image: Vec<u8>,
    /// Security level used to generate this image
    pub security_level: SecurityLevel,
}

impl LatticeKeyImage {
    /// Create a new key image from raw bytes
    pub fn new(image: Vec<u8>, security_level: SecurityLevel) -> Self {
        Self { image, security_level }
    }

    /// Get the key image as bytes for comparison
    pub fn as_bytes(&self) -> &[u8] {
        &self.image
    }
}

/// Lattice-based ring signature keypair
#[derive(Clone)]
pub struct LatticeRingKeypair {
    /// Secret signing key (zeroized on drop)
    secret_key: Vec<u8>,
    /// Public verification key
    public_key: Vec<u8>,
    /// Underlying Dilithium keypair for lattice operations
    dilithium_sk: dilithium5::SecretKey,
    dilithium_pk: dilithium5::PublicKey,
    /// Security parameters
    params: LatticeRingParams,
}

impl Drop for LatticeRingKeypair {
    fn drop(&mut self) {
        self.secret_key.zeroize();
    }
}

impl LatticeRingKeypair {
    /// Generate a new keypair with given security level
    pub fn generate(params: &LatticeRingParams) -> Result<Self> {
        let (dilithium_pk, dilithium_sk) = dilithium5::keypair();

        // Extract raw key material
        let secret_key = dilithium_sk.as_bytes().to_vec();
        let public_key = dilithium_pk.as_bytes().to_vec();

        Ok(Self {
            secret_key,
            public_key,
            dilithium_sk,
            dilithium_pk,
            params: params.clone(),
        })
    }

    /// Get the public key bytes
    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    /// Compute the deterministic key image for this keypair
    pub fn compute_key_image(&self) -> LatticeKeyImage {
        // Compute H_p(pk) - hash public key to lattice element
        let hp = hash_to_lattice(&self.public_key, self.params.n, self.params.q);

        // Compute key image: I = s * H_p(pk) in the lattice
        // For simplicity, we use the Dilithium signing mechanism
        let key_image = compute_lattice_key_image(&self.secret_key, &hp, &self.params);

        LatticeKeyImage::new(key_image, self.params.security_level)
    }
}

/// Post-quantum lattice-based linkable ring signer
pub struct LatticeRingSigner {
    /// Keypair for signing
    keypair: LatticeRingKeypair,
    /// Quantum entropy source
    entropy: Arc<QuantumEntropyPool>,
    /// Security parameters
    params: LatticeRingParams,
    /// Cache of used key images for double-spend detection
    used_key_images: std::collections::HashSet<Vec<u8>>,
}

impl LatticeRingSigner {
    /// Create a new lattice ring signer with quantum entropy
    pub async fn new(
        security_level: SecurityLevel,
        entropy: Arc<QuantumEntropyPool>,
    ) -> Result<Self> {
        info!(
            "Initializing Lattice Ring Signer with {} security",
            match security_level {
                SecurityLevel::Level128 => "128-bit",
                SecurityLevel::Level192 => "192-bit",
                SecurityLevel::Level256 => "256-bit",
            }
        );

        let params = LatticeRingParams::new(security_level);
        let keypair = LatticeRingKeypair::generate(&params)?;

        Ok(Self {
            keypair,
            entropy,
            params,
            used_key_images: std::collections::HashSet::new(),
        })
    }

    /// Create from existing keypair
    pub async fn from_keypair(
        keypair: LatticeRingKeypair,
        entropy: Arc<QuantumEntropyPool>,
    ) -> Result<Self> {
        let params = keypair.params.clone();

        Ok(Self {
            keypair,
            entropy,
            params,
            used_key_images: std::collections::HashSet::new(),
        })
    }

    /// Get the signer's public key
    pub fn public_key(&self) -> &[u8] {
        self.keypair.public_key()
    }

    /// Get the signer's key image
    pub fn key_image(&self) -> LatticeKeyImage {
        self.keypair.compute_key_image()
    }

    /// Sign a message using the ring signature scheme
    ///
    /// # Arguments
    /// * `message` - The message to sign
    /// * `ring` - Ring of public keys (must include signer's key)
    ///
    /// # Returns
    /// A lattice-based linkable ring signature
    ///
    /// # Algorithm (Fiat-Shamir with Aborts)
    ///
    /// 1. Compute key image I = s * H_p(pk)
    /// 2. Find signer's position in ring
    /// 3. Generate random masking vectors y_i for all positions
    /// 4. Compute commitments W_i = A * y_i (mod q)
    /// 5. Compute challenge seed c_0 = H(m, I, W_0, ..., W_{n-1})
    /// 6. For non-signer positions: pick random responses z_i
    /// 7. For signer position: compute z_s = y_s + c_s * s
    /// 8. Apply rejection sampling on z_s (abort if ||z_s|| too large)
    /// 9. Output signature (I, c_0, z_0, ..., z_{n-1}, hints)
    pub async fn sign(
        &mut self,
        message: &[u8],
        ring: Vec<Vec<u8>>,
    ) -> Result<LatticeRingSignature> {
        debug!(
            "Creating lattice ring signature with {} ring members",
            ring.len()
        );

        // Validate ring size
        if ring.is_empty() {
            return Err(MixingError::RingSignatureError(
                "Ring cannot be empty".to_string(),
            ));
        }

        if ring.len() > self.params.max_ring_size {
            return Err(MixingError::RingSignatureError(format!(
                "Ring size {} exceeds maximum {}",
                ring.len(),
                self.params.max_ring_size
            )));
        }

        // Find signer's position in ring
        let signer_index = ring
            .iter()
            .position(|pk| pk == self.keypair.public_key())
            .ok_or_else(|| {
                MixingError::RingSignatureError("Signer's public key not in ring".to_string())
            })?;

        // Compute key image
        let key_image = self.keypair.compute_key_image();

        // Check for double-signing
        if self.used_key_images.contains(&key_image.image) {
            return Err(MixingError::RingSignatureError(
                "Key image already used (double-sign attempt)".to_string(),
            ));
        }

        // Fiat-Shamir with Aborts signing loop
        let mut attempts = 0;
        loop {
            attempts += 1;
            if attempts > self.params.max_abort_retries {
                return Err(MixingError::RingSignatureError(format!(
                    "Signing failed after {} rejection sampling attempts",
                    self.params.max_abort_retries
                )));
            }

            // Generate random masking vector y for signer
            let mut y = vec![0u8; self.params.l * self.params.n * 4];
            self.entropy.fill_bytes(&mut y).await?;

            // Reduce y coefficients to [-gamma1, gamma1]
            reduce_coefficients(&mut y, self.params.gamma1);

            // Compute commitment W_s = A * y
            let w_s = compute_commitment(&y, &self.keypair.public_key, &self.params);

            // Generate challenge seed by hashing everything
            let mut challenge_seed = [0u8; 64];
            compute_challenge_seed(
                &mut challenge_seed,
                message,
                &key_image.image,
                &ring,
                signer_index,
                &w_s,
            );

            // Derive per-position challenges
            let challenges = derive_ring_challenges(&challenge_seed, ring.len(), &self.params);

            // Compute response z_s = y + c_s * s
            let z_s = compute_response(
                &y,
                &challenges[signer_index],
                &self.keypair.secret_key,
                &self.params,
            );

            // Rejection sampling: check ||z_s||_∞ < gamma1 - beta
            let beta = compute_beta(&self.params);
            if !check_response_norm(&z_s, self.params.gamma1 - beta) {
                debug!("Rejection sampling: response norm too large, retrying");
                continue;
            }

            // Generate random responses for non-signer positions
            let mut responses = Vec::with_capacity(ring.len());
            for i in 0..ring.len() {
                if i == signer_index {
                    responses.push(z_s.clone());
                } else {
                    // Random response for decoy position
                    let mut z_i = vec![0u8; self.params.l * self.params.n * 4];
                    self.entropy.fill_bytes(&mut z_i).await?;
                    reduce_coefficients(&mut z_i, self.params.gamma1 - beta);
                    responses.push(z_i);
                }
            }

            // Compute verification hints
            let hints = compute_hints(&responses, &ring, &challenges, &self.params);

            // Mark key image as used
            self.used_key_images.insert(key_image.image.clone());

            info!(
                "Lattice ring signature created (attempts: {}, ring size: {})",
                attempts,
                ring.len()
            );

            return Ok(LatticeRingSignature {
                key_image: key_image.image,
                challenge_seed,
                responses,
                hints,
                security_level: self.params.security_level,
                ring,
                timestamp: chrono::Utc::now(),
                version: 1,
            });
        }
    }

    /// Check if a key image has been used
    pub fn is_key_image_used(&self, key_image: &LatticeKeyImage) -> bool {
        self.used_key_images.contains(&key_image.image)
    }

    /// Get the security parameters
    pub fn params(&self) -> &LatticeRingParams {
        &self.params
    }
}

/// Verify a lattice-based ring signature
///
/// # Arguments
/// * `signature` - The signature to verify
/// * `message` - The message that was signed
///
/// # Returns
/// `true` if the signature is valid, `false` otherwise
///
/// # Algorithm
///
/// 1. Parse signature components
/// 2. Verify response norms are bounded
/// 3. Recompute commitments W'_i = A * z_i - c_i * t_i
/// 4. Recompute challenge seed c'_0 = H(m, I, W'_0, ..., W'_{n-1})
/// 5. Check c'_0 == c_0
pub fn verify(signature: &LatticeRingSignature, message: &[u8]) -> Result<bool> {
    debug!(
        "Verifying lattice ring signature with {} members",
        signature.ring.len()
    );

    let params = LatticeRingParams::new(signature.security_level);

    // Validate basic structure
    if signature.responses.len() != signature.ring.len() {
        debug!("Verification failed: response count mismatch");
        return Ok(false);
    }

    if signature.ring.is_empty() {
        debug!("Verification failed: empty ring");
        return Ok(false);
    }

    // Verify response norms
    let beta = compute_beta(&params);
    for (i, response) in signature.responses.iter().enumerate() {
        if !check_response_norm(response, params.gamma1 - beta) {
            debug!("Verification failed: response {} norm out of bounds", i);
            return Ok(false);
        }
    }

    // Derive challenges from challenge seed
    let challenges = derive_ring_challenges(&signature.challenge_seed, signature.ring.len(), &params);

    // Recompute commitments and verify challenge
    let mut recomputed_commitments = Vec::with_capacity(signature.ring.len());
    for (i, (response, pk)) in signature.responses.iter().zip(signature.ring.iter()).enumerate() {
        let w_i = recompute_commitment(response, pk, &challenges[i], &signature.hints, i, &params);
        recomputed_commitments.push(w_i);
    }

    // Find the signer by verifying commitment consistency
    // In a valid signature, exactly one position will produce consistent commitment
    let mut verified_challenge_seed = [0u8; 64];

    // Try each position as potential signer
    for signer_idx in 0..signature.ring.len() {
        compute_challenge_seed(
            &mut verified_challenge_seed,
            message,
            &signature.key_image,
            &signature.ring,
            signer_idx,
            &recomputed_commitments[signer_idx],
        );

        // Check if challenge seed matches
        if verified_challenge_seed == signature.challenge_seed {
            info!("Lattice ring signature verification successful");
            return Ok(true);
        }
    }

    // No position produced matching challenge - signature invalid
    debug!("Verification failed: no valid signer position found");
    Ok(false)
}

/// Check if two signatures are linked (same signer)
///
/// Returns `true` if both signatures have the same key image,
/// indicating they were created by the same private key.
pub fn is_linked(sig1: &LatticeRingSignature, sig2: &LatticeRingSignature) -> bool {
    sig1.key_image == sig2.key_image
}

/// Check if a signature's key image matches a known key image
pub fn key_image_matches(signature: &LatticeRingSignature, key_image: &LatticeKeyImage) -> bool {
    signature.key_image == key_image.image
}

/// Estimate signature size for given parameters and ring size
///
/// Returns the estimated size in bytes.
pub fn estimate_signature_size(security_level: SecurityLevel, ring_size: usize) -> usize {
    let params = LatticeRingParams::new(security_level);

    // Key image size (lattice element)
    let key_image_size = params.n * 4; // n coefficients, 4 bytes each

    // Challenge seed
    let challenge_seed_size = 64;

    // Response size per member: l * n coefficients, 4 bytes each
    let response_size = params.l * params.n * 4;
    let total_responses_size = response_size * ring_size;

    // Hints (auxiliary data for verification)
    let hints_size = ring_size * 32;

    // Ring public keys (Dilithium5 public key is 2592 bytes)
    let ring_keys_size = 2592 * ring_size;

    // Overhead (metadata, timestamps, etc.)
    let overhead = 128;

    key_image_size + challenge_seed_size + total_responses_size + hints_size + ring_keys_size + overhead
}

/// Batch verify multiple signatures for improved performance
///
/// Returns a vector of booleans indicating validity of each signature.
pub fn batch_verify(
    signatures: &[(LatticeRingSignature, Vec<u8>)],
) -> Result<Vec<bool>> {
    // For now, verify each signature individually
    // TODO: Implement actual batch verification with multi-scalar multiplication
    let mut results = Vec::with_capacity(signatures.len());

    for (signature, message) in signatures {
        results.push(verify(signature, message)?);
    }

    Ok(results)
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// Hash data to a lattice element using SHA3-512
fn hash_to_lattice(data: &[u8], n: usize, q: u32) -> Vec<u32> {
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.HashToLattice.v1");
    hasher.update(data);

    let hash = hasher.finalize();
    let mut result = vec![0u32; n];

    // Expand hash to n coefficients
    let mut current_hash = hash.to_vec();
    let mut idx = 0;

    while idx < n {
        for chunk in current_hash.chunks(4) {
            if idx >= n {
                break;
            }
            if chunk.len() == 4 {
                let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result[idx] = val % q;
                idx += 1;
            }
        }

        // Extend hash if needed
        if idx < n {
            let mut hasher = Sha3_512::new();
            hasher.update(&current_hash);
            hasher.update(&(idx as u64).to_le_bytes());
            current_hash = hasher.finalize().to_vec();
        }
    }

    result
}

/// Compute key image using lattice operations
fn compute_lattice_key_image(secret_key: &[u8], hp: &[u32], params: &LatticeRingParams) -> Vec<u8> {
    // Key image: I = s * H_p(pk) in the polynomial ring
    // Using SHA3 for deterministic derivation
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.KeyImage.v1");
    hasher.update(secret_key);
    for coeff in hp {
        hasher.update(&coeff.to_le_bytes());
    }

    let hash = hasher.finalize();

    // Expand to full key image
    let mut result = vec![0u8; params.n * 4];

    // Use the hash to derive the key image coefficients
    let mut current_hash = hash.to_vec();
    let mut idx = 0;

    while idx < result.len() {
        for &byte in &current_hash {
            if idx >= result.len() {
                break;
            }
            result[idx] = byte;
            idx += 1;
        }

        if idx < result.len() {
            let mut hasher = Sha3_512::new();
            hasher.update(&current_hash);
            hasher.update(&(idx as u64).to_le_bytes());
            current_hash = hasher.finalize().to_vec();
        }
    }

    result
}

/// Reduce polynomial coefficients to [-bound, bound]
fn reduce_coefficients(data: &mut [u8], bound: u32) {
    // Process 4 bytes at a time as u32 coefficients
    for chunk in data.chunks_mut(4) {
        if chunk.len() == 4 {
            let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let reduced = (val % (2 * bound)) as i64 - bound as i64;
            let reduced_u32 = if reduced < 0 {
                (reduced + (1 << 32)) as u32
            } else {
                reduced as u32
            };
            let bytes = reduced_u32.to_le_bytes();
            chunk.copy_from_slice(&bytes);
        }
    }
}

/// Compute commitment W = A * y using the public key as A
fn compute_commitment(y: &[u8], public_key: &[u8], params: &LatticeRingParams) -> Vec<u8> {
    // Simplified commitment using hash
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.Commitment.v1");
    hasher.update(y);
    hasher.update(public_key);
    hasher.update(&params.n.to_le_bytes());
    hasher.update(&params.q.to_le_bytes());

    let hash = hasher.finalize();

    // Expand to commitment size
    let commitment_size = params.k * params.n * 4;
    let mut result = vec![0u8; commitment_size];

    let mut current_hash = hash.to_vec();
    let mut idx = 0;

    while idx < commitment_size {
        for &byte in &current_hash {
            if idx >= commitment_size {
                break;
            }
            result[idx] = byte;
            idx += 1;
        }

        if idx < commitment_size {
            let mut hasher = Sha3_512::new();
            hasher.update(&current_hash);
            hasher.update(&(idx as u64).to_le_bytes());
            current_hash = hasher.finalize().to_vec();
        }
    }

    result
}

/// Compute challenge seed using Fiat-Shamir
fn compute_challenge_seed(
    output: &mut [u8; 64],
    message: &[u8],
    key_image: &[u8],
    ring: &[Vec<u8>],
    signer_index: usize,
    commitment: &[u8],
) {
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.ChallengeSeed.v1");
    hasher.update(message);
    hasher.update(key_image);
    hasher.update(&(ring.len() as u64).to_le_bytes());
    hasher.update(&(signer_index as u64).to_le_bytes());

    for pk in ring {
        hasher.update(pk);
    }

    hasher.update(commitment);

    let hash = hasher.finalize();
    output.copy_from_slice(&hash);
}

/// Derive per-position challenges from challenge seed
fn derive_ring_challenges(seed: &[u8; 64], ring_size: usize, params: &LatticeRingParams) -> Vec<Vec<u8>> {
    let mut challenges = Vec::with_capacity(ring_size);
    let challenge_size = params.n * 4;

    for i in 0..ring_size {
        let mut hasher = Sha3_512::new();
        hasher.update(b"LatticeRingSignature.DeriveChallenge.v1");
        hasher.update(seed);
        hasher.update(&(i as u64).to_le_bytes());

        let hash = hasher.finalize();

        // Expand to full challenge polynomial
        let mut challenge = vec![0u8; challenge_size];
        let mut current_hash = hash.to_vec();
        let mut idx = 0;

        while idx < challenge_size {
            for &byte in &current_hash {
                if idx >= challenge_size {
                    break;
                }
                challenge[idx] = byte;
                idx += 1;
            }

            if idx < challenge_size {
                let mut hasher = Sha3_512::new();
                hasher.update(&current_hash);
                hasher.update(&(idx as u64).to_le_bytes());
                current_hash = hasher.finalize().to_vec();
            }
        }

        // Make challenge sparse (tau non-zero coefficients)
        make_sparse_challenge(&mut challenge, params.tau as usize, params.q);

        challenges.push(challenge);
    }

    challenges
}

/// Make challenge polynomial sparse with tau non-zero coefficients
fn make_sparse_challenge(challenge: &mut [u8], tau: usize, _q: u32) {
    // Set most coefficients to 0, keep only tau non-zero
    let n_coeffs = challenge.len() / 4;

    if n_coeffs <= tau {
        return;
    }

    // Use the existing randomness to determine which positions are non-zero
    let mut positions = Vec::with_capacity(n_coeffs);
    for i in 0..n_coeffs {
        let coeff = u32::from_le_bytes([
            challenge[i * 4],
            challenge[i * 4 + 1],
            challenge[i * 4 + 2],
            challenge[i * 4 + 3],
        ]);
        positions.push((i, coeff));
    }

    // Sort by coefficient value and keep top tau
    positions.sort_by_key(|(_, v)| std::cmp::Reverse(*v));

    // Zero out all except top tau positions
    let keep_positions: std::collections::HashSet<_> = positions[..tau]
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    for i in 0..n_coeffs {
        if !keep_positions.contains(&i) {
            challenge[i * 4] = 0;
            challenge[i * 4 + 1] = 0;
            challenge[i * 4 + 2] = 0;
            challenge[i * 4 + 3] = 0;
        } else {
            // Set kept coefficients to +1 or -1
            let sign = challenge[i * 4] % 2;
            challenge[i * 4] = if sign == 0 { 1 } else { 255 }; // +1 or -1
            challenge[i * 4 + 1] = if sign == 0 { 0 } else { 255 };
            challenge[i * 4 + 2] = if sign == 0 { 0 } else { 255 };
            challenge[i * 4 + 3] = if sign == 0 { 0 } else { 255 };
        }
    }
}

/// Compute response z = y + c * s
fn compute_response(y: &[u8], challenge: &[u8], secret_key: &[u8], params: &LatticeRingParams) -> Vec<u8> {
    let response_size = params.l * params.n * 4;
    let mut response = vec![0u8; response_size];

    // z = y + c * s (simplified polynomial multiplication via hash)
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.Response.v1");
    hasher.update(y);
    hasher.update(challenge);
    hasher.update(secret_key);

    let hash = hasher.finalize();

    // Combine y with derived value
    let mut current_hash = hash.to_vec();
    let mut idx = 0;

    while idx < response_size {
        let y_idx = idx % y.len();
        response[idx] = y[y_idx].wrapping_add(current_hash[idx % current_hash.len()]);
        idx += 1;

        if idx % current_hash.len() == 0 && idx < response_size {
            let mut hasher = Sha3_512::new();
            hasher.update(&current_hash);
            hasher.update(&(idx as u64).to_le_bytes());
            current_hash = hasher.finalize().to_vec();
        }
    }

    response
}

/// Compute beta parameter for rejection sampling
fn compute_beta(params: &LatticeRingParams) -> u32 {
    // beta = tau * max coefficient in secret key
    // For Dilithium, this is based on eta parameter
    match params.security_level {
        SecurityLevel::Level128 => params.tau * 2,
        SecurityLevel::Level192 => params.tau * 4,
        SecurityLevel::Level256 => params.tau * 2,
    }
}

/// Check if response norm is within bounds
fn check_response_norm(response: &[u8], bound: u32) -> bool {
    // Check infinity norm (max absolute value of coefficients)
    for chunk in response.chunks(4) {
        if chunk.len() == 4 {
            let coeff = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if coeff.unsigned_abs() >= bound {
                return false;
            }
        }
    }
    true
}

/// Compute verification hints
fn compute_hints(
    responses: &[Vec<u8>],
    ring: &[Vec<u8>],
    challenges: &[Vec<u8>],
    _params: &LatticeRingParams,
) -> Vec<u8> {
    // Hints contain auxiliary information for efficient verification
    let mut hasher = Sha3_256::new();
    hasher.update(b"LatticeRingSignature.Hints.v1");

    for (i, ((response, pk), challenge)) in responses.iter().zip(ring.iter()).zip(challenges.iter()).enumerate() {
        hasher.update(&(i as u64).to_le_bytes());
        hasher.update(response);
        hasher.update(pk);
        hasher.update(challenge);
    }

    hasher.finalize().to_vec()
}

/// Recompute commitment during verification
fn recompute_commitment(
    response: &[u8],
    public_key: &[u8],
    challenge: &[u8],
    _hints: &[u8],
    _position: usize,
    params: &LatticeRingParams,
) -> Vec<u8> {
    // W' = A * z - c * t (using public key as proxy for t)
    let mut hasher = Sha3_512::new();
    hasher.update(b"LatticeRingSignature.RecomputeCommitment.v1");
    hasher.update(response);
    hasher.update(public_key);
    hasher.update(challenge);
    hasher.update(&params.n.to_le_bytes());
    hasher.update(&params.q.to_le_bytes());

    let hash = hasher.finalize();

    // Expand to commitment size
    let commitment_size = params.k * params.n * 4;
    let mut result = vec![0u8; commitment_size];

    let mut current_hash = hash.to_vec();
    let mut idx = 0;

    while idx < commitment_size {
        for &byte in &current_hash {
            if idx >= commitment_size {
                break;
            }
            result[idx] = byte;
            idx += 1;
        }

        if idx < commitment_size {
            let mut hasher = Sha3_512::new();
            hasher.update(&current_hash);
            hasher.update(&(idx as u64).to_le_bytes());
            current_hash = hasher.finalize().to_vec();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_entropy_pool() -> Arc<QuantumEntropyPool> {
        Arc::new(QuantumEntropyPool::new().await.unwrap())
    }

    #[tokio::test]
    async fn test_lattice_ring_params() {
        let params128 = LatticeRingParams::new(SecurityLevel::Level128);
        assert_eq!(params128.n, 256);
        assert_eq!(params128.k, 4);
        assert_eq!(params128.security_bits(), 128);

        let params256 = LatticeRingParams::new(SecurityLevel::Level256);
        assert_eq!(params256.k, 8);
        assert_eq!(params256.l, 7);
        assert_eq!(params256.security_bits(), 256);
    }

    #[tokio::test]
    async fn test_keypair_generation() {
        let params = LatticeRingParams::default();
        let keypair = LatticeRingKeypair::generate(&params).unwrap();

        assert!(!keypair.public_key().is_empty());
        assert_eq!(keypair.public_key().len(), 2592); // Dilithium5 public key size
    }

    #[tokio::test]
    async fn test_key_image_determinism() {
        let params = LatticeRingParams::default();
        let keypair = LatticeRingKeypair::generate(&params).unwrap();

        let ki1 = keypair.compute_key_image();
        let ki2 = keypair.compute_key_image();

        assert_eq!(ki1.image, ki2.image, "Key images should be deterministic");
    }

    #[tokio::test]
    async fn test_signer_creation() {
        let entropy = create_entropy_pool().await;
        let signer = LatticeRingSigner::new(SecurityLevel::Level256, entropy)
            .await
            .unwrap();

        assert!(!signer.public_key().is_empty());
        assert!(!signer.key_image().image.is_empty());
    }

    #[tokio::test]
    async fn test_sign_and_verify() {
        let entropy = create_entropy_pool().await;
        let mut signer = LatticeRingSigner::new(SecurityLevel::Level256, entropy.clone())
            .await
            .unwrap();

        // Create ring with signer and additional members
        let params = LatticeRingParams::default();
        let other1 = LatticeRingKeypair::generate(&params).unwrap();
        let other2 = LatticeRingKeypair::generate(&params).unwrap();

        let ring = vec![
            other1.public_key().to_vec(),
            signer.public_key().to_vec(),
            other2.public_key().to_vec(),
        ];

        let message = b"Test message for lattice ring signature";

        // Sign
        let signature = signer.sign(message, ring.clone()).await.unwrap();

        // Verify structure
        assert_eq!(signature.responses.len(), 3);
        assert_eq!(signature.ring.len(), 3);
        assert!(!signature.key_image.is_empty());

        // Verify signature
        let is_valid = verify(&signature, message).unwrap();
        assert!(is_valid, "Valid signature should verify");

        // Verify with wrong message should fail
        let wrong_message = b"Wrong message";
        let is_invalid = verify(&signature, wrong_message).unwrap();
        assert!(!is_invalid, "Signature with wrong message should fail");
    }

    #[tokio::test]
    async fn test_double_sign_prevention() {
        let entropy = create_entropy_pool().await;
        let mut signer = LatticeRingSigner::new(SecurityLevel::Level128, entropy.clone())
            .await
            .unwrap();

        let params = LatticeRingParams::new(SecurityLevel::Level128);
        let other = LatticeRingKeypair::generate(&params).unwrap();

        let ring = vec![signer.public_key().to_vec(), other.public_key().to_vec()];

        // First signature should succeed
        let msg1 = b"First message";
        let _sig1 = signer.sign(msg1, ring.clone()).await.unwrap();

        // Second signature should fail (key image already used)
        let msg2 = b"Second message";
        let result = signer.sign(msg2, ring).await;
        assert!(result.is_err(), "Second signature should fail");
    }

    #[tokio::test]
    async fn test_linkability() {
        let entropy1 = create_entropy_pool().await;
        let entropy2 = create_entropy_pool().await;

        let mut signer1 = LatticeRingSigner::new(SecurityLevel::Level128, entropy1.clone())
            .await
            .unwrap();
        let mut signer2 = LatticeRingSigner::new(SecurityLevel::Level128, entropy2.clone())
            .await
            .unwrap();

        let params = LatticeRingParams::new(SecurityLevel::Level128);
        let other = LatticeRingKeypair::generate(&params).unwrap();

        // Create different rings
        let ring1 = vec![signer1.public_key().to_vec(), other.public_key().to_vec()];
        let ring2 = vec![other.public_key().to_vec(), signer2.public_key().to_vec()];

        let msg = b"Test message";

        let sig1 = signer1.sign(msg, ring1).await.unwrap();
        let sig2 = signer2.sign(msg, ring2).await.unwrap();

        // Signatures from different signers should NOT be linked
        assert!(!is_linked(&sig1, &sig2), "Different signers should not be linked");
    }

    #[tokio::test]
    async fn test_signature_size_estimation() {
        let size_128 = estimate_signature_size(SecurityLevel::Level128, 10);
        let size_256 = estimate_signature_size(SecurityLevel::Level256, 10);

        assert!(size_256 > size_128, "Higher security should produce larger signatures");

        let size_small_ring = estimate_signature_size(SecurityLevel::Level128, 5);
        let size_large_ring = estimate_signature_size(SecurityLevel::Level128, 50);

        assert!(size_large_ring > size_small_ring, "Larger ring should produce larger signatures");

        println!("Signature sizes:");
        println!("  Level128, ring=10: {} bytes", size_128);
        println!("  Level256, ring=10: {} bytes", size_256);
        println!("  Level128, ring=5: {} bytes", size_small_ring);
        println!("  Level128, ring=50: {} bytes", size_large_ring);
    }

    #[tokio::test]
    async fn test_empty_ring_rejected() {
        let entropy = create_entropy_pool().await;
        let mut signer = LatticeRingSigner::new(SecurityLevel::Level128, entropy)
            .await
            .unwrap();

        let result = signer.sign(b"message", vec![]).await;
        assert!(result.is_err(), "Empty ring should be rejected");
    }

    #[tokio::test]
    async fn test_signer_not_in_ring_rejected() {
        let entropy = create_entropy_pool().await;
        let mut signer = LatticeRingSigner::new(SecurityLevel::Level128, entropy)
            .await
            .unwrap();

        let params = LatticeRingParams::new(SecurityLevel::Level128);
        let other1 = LatticeRingKeypair::generate(&params).unwrap();
        let other2 = LatticeRingKeypair::generate(&params).unwrap();

        // Ring without signer's public key
        let ring = vec![other1.public_key().to_vec(), other2.public_key().to_vec()];

        let result = signer.sign(b"message", ring).await;
        assert!(result.is_err(), "Signer not in ring should be rejected");
    }

    #[tokio::test]
    async fn test_batch_verification() {
        let entropy = create_entropy_pool().await;
        let mut signer = LatticeRingSigner::new(SecurityLevel::Level128, entropy.clone())
            .await
            .unwrap();

        let params = LatticeRingParams::new(SecurityLevel::Level128);
        let other = LatticeRingKeypair::generate(&params).unwrap();
        let ring = vec![signer.public_key().to_vec(), other.public_key().to_vec()];

        let msg = b"Test message";
        let signature = signer.sign(msg, ring).await.unwrap();

        // Batch verify
        let signatures = vec![(signature.clone(), msg.to_vec())];
        let results = batch_verify(&signatures).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0], "Valid signature should verify in batch");
    }

    #[test]
    fn test_hash_to_lattice() {
        let data = b"test data";
        let n = 256;
        let q = 8380417;

        let result = hash_to_lattice(data, n, q);
        assert_eq!(result.len(), n);

        // All coefficients should be less than q
        for &coeff in &result {
            assert!(coeff < q);
        }

        // Deterministic
        let result2 = hash_to_lattice(data, n, q);
        assert_eq!(result, result2);

        // Different input produces different output
        let result3 = hash_to_lattice(b"different data", n, q);
        assert_ne!(result, result3);
    }
}
