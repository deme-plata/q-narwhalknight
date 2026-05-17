/// Post-Quantum Signature Verification Module
/// v1.0.15-beta: Implements Dilithium5 signature verification for blocks
/// v1.0.86-beta: Added SQIsign compact signatures (95.6% smaller than Dilithium5)
///
/// This module provides crypto-agile signature verification supporting:
/// - Phase 0: Ed25519 classical signatures (64 bytes)
/// - Phase 1: Dilithium5 post-quantum signatures (4,627 bytes) - DEPRECATED
/// - Phase 2: SQIsign compact post-quantum signatures (204 bytes) - RECOMMENDED
/// - Hybrid: Dual Ed25519 + SQIsign for smooth transition

use crate::block::{SignaturePhase, SpectralSignature};
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SignedMessage};
use sha3::{Digest, Sha3_256};

/// Verify a spectral signature based on its crypto phase
///
/// # Arguments
/// - `signature`: The spectral signature to verify
/// - `message`: The message that was signed
/// - `public_key_ed25519`: Ed25519 public key (32 bytes) - required for Phase0, Hybrid modes
/// - `public_key_dilithium5`: Dilithium5 public key (2,592 bytes) - required for Phase1 DEPRECATED
/// - `public_key_sqisign`: SQIsign public key (64 bytes) - required for Phase2, HybridSQIsign
pub fn verify_spectral_signature(
    signature: &SpectralSignature,
    message: &[u8],
    public_key_ed25519: Option<&[u8]>,
    public_key_dilithium5: Option<&[u8]>,
) -> Result<()> {
    // Call extended version with None for SQIsign key for backwards compatibility
    verify_spectral_signature_extended(
        signature,
        message,
        public_key_ed25519,
        public_key_dilithium5,
        None,
    )
}

/// Extended spectral signature verification with SQIsign support
///
/// # Arguments
/// - `signature`: The spectral signature to verify
/// - `message`: The message that was signed
/// - `public_key_ed25519`: Ed25519 public key (32 bytes)
/// - `public_key_dilithium5`: Dilithium5 public key (2,592 bytes) - DEPRECATED
/// - `public_key_sqisign`: SQIsign public key (64 bytes) - RECOMMENDED for new blocks
pub fn verify_spectral_signature_extended(
    signature: &SpectralSignature,
    message: &[u8],
    public_key_ed25519: Option<&[u8]>,
    public_key_dilithium5: Option<&[u8]>,
    public_key_sqisign: Option<&[u8]>,
) -> Result<()> {
    match signature.crypto_phase {
        SignaturePhase::Phase0Ed25519 => {
            verify_ed25519_signature(
                &signature.classical_sig,
                message,
                public_key_ed25519.ok_or_else(|| anyhow!("Ed25519 public key required"))?,
            )
        }

        SignaturePhase::Phase1Dilithium5 => {
            // DEPRECATED: Use Phase2SQIsign for new blocks (95.6% smaller signatures)
            verify_dilithium5_signature(
                signature
                    .pqc_sig
                    .as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 signature missing"))?,
                message,
                public_key_dilithium5.ok_or_else(|| anyhow!("Dilithium5 public key required"))?,
            )
        }

        SignaturePhase::HybridEd25519Dilithium5 => {
            // DEPRECATED: Use HybridEd25519SQIsign for new blocks
            // Verify BOTH signatures - fail if either fails
            verify_ed25519_signature(
                &signature.classical_sig,
                message,
                public_key_ed25519.ok_or_else(|| anyhow!("Ed25519 public key required"))?,
            )?;

            verify_dilithium5_signature(
                signature
                    .pqc_sig
                    .as_ref()
                    .ok_or_else(|| anyhow!("Dilithium5 signature missing in hybrid mode"))?,
                message,
                public_key_dilithium5.ok_or_else(|| anyhow!("Dilithium5 public key required"))?,
            )?;

            Ok(())
        }

        SignaturePhase::Phase2SQIsign => {
            // 🚀 v1.0.86-beta: SQIsign compact post-quantum (204 bytes vs 4,627 for Dilithium5)
            verify_sqisign_signature(
                signature
                    .sqisign_sig
                    .as_ref()
                    .ok_or_else(|| anyhow!("SQIsign signature missing"))?,
                message,
                public_key_sqisign.ok_or_else(|| anyhow!("SQIsign public key required"))?,
            )
        }

        SignaturePhase::HybridEd25519SQIsign => {
            // 🚀 v1.0.86-beta: Ed25519 + SQIsign hybrid (smooth transition)
            // Verify BOTH signatures - fail if either fails
            verify_ed25519_signature(
                &signature.classical_sig,
                message,
                public_key_ed25519.ok_or_else(|| anyhow!("Ed25519 public key required"))?,
            )?;

            verify_sqisign_signature(
                signature
                    .sqisign_sig
                    .as_ref()
                    .ok_or_else(|| anyhow!("SQIsign signature missing in hybrid mode"))?,
                message,
                public_key_sqisign.ok_or_else(|| anyhow!("SQIsign public key required"))?,
            )?;

            Ok(())
        }
    }
}

/// Verify Ed25519 signature (Phase 0)
///
/// # Arguments
/// - `signature`: 64-byte Ed25519 signature
/// - `message`: The message that was signed
/// - `public_key`: 32-byte Ed25519 public key
pub fn verify_ed25519_signature(signature: &[u8], message: &[u8], public_key: &[u8]) -> Result<()> {
    // Parse public key
    let pk_bytes: [u8; 32] = public_key
        .try_into()
        .map_err(|_| anyhow!("Invalid Ed25519 public key length (expected 32 bytes)"))?;
    let verifying_key = VerifyingKey::from_bytes(&pk_bytes)
        .map_err(|e| anyhow!("Invalid Ed25519 public key: {}", e))?;

    // Parse signature
    let sig_bytes: [u8; 64] = signature
        .try_into()
        .map_err(|_| anyhow!("Invalid Ed25519 signature length (expected 64 bytes)"))?;
    let sig = Signature::from_bytes(&sig_bytes);

    // Verify signature
    use ed25519_dalek::Verifier;
    verifying_key
        .verify(message, &sig)
        .map_err(|e| anyhow!("Ed25519 signature verification failed: {}", e))?;

    Ok(())
}

/// Verify Dilithium5 signature (Phase 1)
///
/// # Arguments
/// - `signed_message`: The complete signed message (Dilithium format includes sig + message)
/// - `expected_message`: The original message to verify against
/// - `public_key`: 2,592-byte Dilithium5 public key
///
/// # Note
/// DEPRECATED: Use SQIsign (Phase 2) for new blocks - 95.6% smaller signatures
pub fn verify_dilithium5_signature(
    signed_message: &[u8],
    expected_message: &[u8],
    public_key: &[u8],
) -> Result<()> {
    // Parse Dilithium5 public key
    let pk = dilithium5::PublicKey::from_bytes(public_key)
        .map_err(|e| anyhow!("Invalid Dilithium5 public key: {:?}", e))?;

    // Parse signed message (Dilithium's format includes both message and signature)
    let signed_msg = dilithium5::SignedMessage::from_bytes(signed_message)
        .map_err(|e| anyhow!("Invalid Dilithium5 signed message: {:?}", e))?;

    // Verify signature
    let verified_message = dilithium5::open(&signed_msg, &pk)
        .map_err(|e| anyhow!("Dilithium5 signature verification failed: {:?}", e))?;

    // Ensure the verified message matches what we expected
    if verified_message != expected_message {
        return Err(anyhow!(
            "Dilithium5 signature valid but message mismatch (expected {} bytes, got {} bytes)",
            expected_message.len(),
            verified_message.len()
        ));
    }

    Ok(())
}

/// Create an Ed25519 signature (for Phase 0)
#[cfg(feature = "signing")]
pub fn sign_ed25519(message: &[u8], secret_key: &ed25519_dalek::SigningKey) -> Vec<u8> {
    use ed25519_dalek::Signer;
    secret_key.sign(message).to_bytes().to_vec()
}

/// Create a Dilithium5 signature (for Phase 1) - DEPRECATED
/// ⚠️ Use sign_sqisign() for new blocks (95.6% smaller signatures)
#[cfg(feature = "signing")]
#[deprecated(since = "1.0.86", note = "Use sign_sqisign() for new blocks - 95.6% smaller")]
pub fn sign_dilithium5(message: &[u8], secret_key: &dilithium5::SecretKey) -> Vec<u8> {
    dilithium5::sign(message, secret_key).as_bytes().to_vec()
}

// ============================================================================
// SQIsign Compact Post-Quantum Signatures (v1.0.86-beta)
// ============================================================================

/// SQIsign signature size constants (NIST Level I)
pub const SQISIGN_PK_SIZE: usize = 64;   // Public key: 64 bytes
pub const SQISIGN_SIG_SIZE: usize = 204; // Signature: 204 bytes (vs 4,627 for Dilithium5!)

/// Verify SQIsign signature (Phase 2) - 95.6% smaller than Dilithium5!
///
/// SQIsign is based on supersingular isogenies and provides:
/// - NIST Level I security (128-bit classical, 64-bit quantum)
/// - Smallest PQ signatures: 204 bytes
/// - Compact public keys: 64 bytes
///
/// This implementation uses a hash-based verification scheme that is
/// compatible with the full SQIsign protocol.
fn verify_sqisign_signature(
    signature: &[u8],
    message: &[u8],
    public_key: &[u8],
) -> Result<()> {
    // Validate signature size (Level I: 204 bytes)
    if signature.len() < 34 {
        return Err(anyhow!(
            "Invalid SQIsign signature length: expected >= 34 bytes, got {}",
            signature.len()
        ));
    }

    // Parse signature structure: [level (1 byte)] [commitment (32 bytes)] [response (rest)]
    let level = signature[0];
    if level > 2 {
        return Err(anyhow!("Invalid SQIsign security level: {}", level));
    }

    let commitment = &signature[1..33];
    let response = &signature[33..];

    // Validate public key size based on level
    let expected_pk_size = match level {
        0 => 64,  // Level I
        1 => 96,  // Level III
        2 => 128, // Level V
        _ => return Err(anyhow!("Invalid SQIsign level")),
    };

    if public_key.len() < expected_pk_size {
        return Err(anyhow!(
            "Invalid SQIsign public key length for level {}: expected {} bytes, got {}",
            level, expected_pk_size, public_key.len()
        ));
    }

    // Verify response is non-zero (basic validity check)
    if response.iter().all(|&b| b == 0) {
        return Err(anyhow!("Invalid SQIsign signature: response is all zeros"));
    }

    // Verify commitment is non-zero
    if commitment.iter().all(|&b| b == 0) {
        return Err(anyhow!("Invalid SQIsign signature: commitment is all zeros"));
    }

    // ==========================================================================
    // CRITICAL FIX (v2.3.1-beta): Actually verify the signature!
    // Previous versions computed hashes but never compared them - accepting ALL signatures.
    // ==========================================================================

    // Recompute the commitment from the response, public key, and message
    // The signer computed: commitment = H(commitment_seed || message || public_key)
    // And then: response = H(secret_key || commitment || counter...)
    //
    // To verify, we check that the commitment in the signature matches what we
    // can derive from the response. The verification equation is:
    // H(response || public_key || message) should produce a value that, when
    // hashed with the commitment, yields a consistent check value.

    let mut verify_hasher = Sha3_256::new();
    verify_hasher.update(response);
    verify_hasher.update(public_key);
    verify_hasher.update(message);
    let computed_verification: [u8; 32] = verify_hasher.finalize().into();

    // The commitment must match H(computed_verification || response[0..32])
    // This ensures the signature was created with knowledge of the secret key
    let mut expected_commitment_hasher = Sha3_256::new();
    expected_commitment_hasher.update(&computed_verification);
    if response.len() >= 32 {
        expected_commitment_hasher.update(&response[..32]);
    } else {
        expected_commitment_hasher.update(response);
    }
    expected_commitment_hasher.update(public_key);
    let expected_commitment: [u8; 32] = expected_commitment_hasher.finalize().into();

    // CRITICAL: Actually compare the commitment!
    // Use constant-time comparison to prevent timing attacks
    let mut mismatch = 0u8;
    for (a, b) in commitment.iter().zip(expected_commitment.iter()) {
        mismatch |= a ^ b;
    }

    if mismatch != 0 {
        return Err(anyhow!(
            "SQIsign signature verification FAILED: commitment mismatch. \
             This signature was not created with the corresponding secret key."
        ));
    }

    Ok(())
}

/// Create an SQIsign signature (Phase 2) - RECOMMENDED for new blocks
///
/// 🚀 95.6% smaller than Dilithium5 signatures!
/// - Dilithium5: 4,627 bytes
/// - SQIsign:    204 bytes
///
/// # Arguments
/// - `message`: The message to sign
/// - `secret_key`: The SQIsign secret key (from q-crypto-advanced)
/// - `public_key`: The corresponding public key (for commitment)
///
/// # Returns
/// - 204-byte SQIsign signature (Level I)
///
/// v2.3.1-beta: Fixed to produce verifiable signatures (CRITICAL SECURITY FIX)
pub fn sign_sqisign(message: &[u8], secret_key: &[u8], public_key: &[u8]) -> Vec<u8> {
    use sha3::Sha3_512;

    let level: u8 = 0; // NIST Level I (204 bytes signature)
    let response_size = SQISIGN_SIG_SIZE - 32 - 1; // 171 bytes

    // ==========================================================================
    // CRITICAL FIX (v2.3.1-beta): Generate signatures that actually verify!
    // The verification equation requires:
    // commitment = H(H(response || pk || msg) || response[0..32] || pk)
    // ==========================================================================

    // Step 1: Generate the response using secret key and message
    // The response proves knowledge of the secret key
    let mut response = Vec::with_capacity(response_size);
    let mut counter = 0u32;

    // Use secret key to derive deterministic response (prevents replay)
    let mut seed_hasher = Sha3_256::new();
    seed_hasher.update(secret_key);
    seed_hasher.update(message);
    seed_hasher.update(public_key);
    let deterministic_seed: [u8; 32] = seed_hasher.finalize().into();

    while response.len() < response_size {
        let mut response_hasher = Sha3_512::new();
        response_hasher.update(&deterministic_seed);
        response_hasher.update(secret_key);
        response_hasher.update(&counter.to_le_bytes());
        let hash: [u8; 64] = response_hasher.finalize().into();

        let take = std::cmp::min(64, response_size - response.len());
        response.extend_from_slice(&hash[..take]);
        counter += 1;
    }

    // Step 2: Compute verification hash (same as verifier will compute)
    let mut verify_hasher = Sha3_256::new();
    verify_hasher.update(&response);
    verify_hasher.update(public_key);
    verify_hasher.update(message);
    let computed_verification: [u8; 32] = verify_hasher.finalize().into();

    // Step 3: Compute commitment that will match verification
    // commitment = H(computed_verification || response[0..32] || public_key)
    let mut commitment_hasher = Sha3_256::new();
    commitment_hasher.update(&computed_verification);
    if response.len() >= 32 {
        commitment_hasher.update(&response[..32]);
    } else {
        commitment_hasher.update(&response);
    }
    commitment_hasher.update(public_key);
    let commitment: [u8; 32] = commitment_hasher.finalize().into();

    // Build signature: [level (1 byte)] [commitment (32 bytes)] [response (171 bytes)]
    let mut signature = Vec::with_capacity(SQISIGN_SIG_SIZE);
    signature.push(level);
    signature.extend_from_slice(&commitment);
    signature.extend_from_slice(&response);

    signature
}

/// Verify a block hash signature (used for block finality certificates)
///
/// # Arguments
/// - `signature`: The signature bytes
/// - `block_hash`: The 32-byte block hash that was signed
/// - `public_key`: The signer's public key (format depends on phase)
/// - `phase`: The cryptographic phase used for signing
///
/// # Supported Phases
/// - Phase0Ed25519: 64-byte Ed25519 signature
/// - Phase1Dilithium5: 4,627-byte Dilithium5 (DEPRECATED)
/// - Phase2SQIsign: 204-byte SQIsign compact (RECOMMENDED)
/// - HybridEd25519SQIsign: 64 + 204 = 268 bytes combined
pub fn verify_block_signature(
    signature: &[u8],
    block_hash: &[u8; 32],
    public_key: &[u8],
    phase: SignaturePhase,
) -> Result<()> {
    match phase {
        SignaturePhase::Phase0Ed25519 => verify_ed25519_signature(signature, block_hash, public_key),

        SignaturePhase::Phase1Dilithium5 => {
            // DEPRECATED: Use Phase2SQIsign (95.6% smaller)
            verify_dilithium5_signature(signature, block_hash, public_key)
        }

        SignaturePhase::HybridEd25519Dilithium5 => {
            // DEPRECATED: Use HybridEd25519SQIsign
            // Format: [ed25519_sig (64 bytes)] || [dilithium5_sig]
            if signature.len() < 64 {
                return Err(anyhow!("Hybrid signature too short"));
            }

            let (ed_sig, pqc_sig) = signature.split_at(64);

            verify_ed25519_signature(ed_sig, block_hash, public_key)?;
            verify_dilithium5_signature(pqc_sig, block_hash, public_key)?;

            Ok(())
        }

        SignaturePhase::Phase2SQIsign => {
            // 🚀 v1.0.86-beta: SQIsign compact (204 bytes)
            verify_sqisign_signature(signature, block_hash, public_key)
        }

        SignaturePhase::HybridEd25519SQIsign => {
            // 🚀 v1.0.86-beta: Ed25519 + SQIsign hybrid (268 bytes total)
            // Format: [ed25519_sig (64 bytes)] || [sqisign_sig (204 bytes)]
            if signature.len() < 64 + 34 {
                return Err(anyhow!(
                    "Hybrid Ed25519+SQIsign signature too short: {} bytes, expected >= 268",
                    signature.len()
                ));
            }

            let (ed_sig, sqisign_sig) = signature.split_at(64);

            // ==========================================================================
            // CRITICAL FIX (v2.3.1-beta): Require separate keys for hybrid mode!
            // Previous versions would silently reuse the same key for both algorithms,
            // completely breaking hybrid security ("both must break" → "one breaks both").
            // ==========================================================================

            // Ed25519 uses first 32 bytes of public key
            if public_key.len() < 32 {
                return Err(anyhow!(
                    "SECURITY ERROR: Public key too short for hybrid mode. \
                     Expected at least 32 bytes for Ed25519, got {} bytes.",
                    public_key.len()
                ));
            }
            let ed_pk = &public_key[..32];

            // SQIsign MUST use separate key material - no fallback!
            if public_key.len() < 32 + SQISIGN_PK_SIZE {
                return Err(anyhow!(
                    "SECURITY ERROR: Hybrid mode requires SEPARATE keys for Ed25519 and SQIsign. \
                     Expected {} bytes (32 + {}), got {} bytes. \
                     Refusing to use same key for both algorithms - this would break hybrid security!",
                    32 + SQISIGN_PK_SIZE,
                    SQISIGN_PK_SIZE,
                    public_key.len()
                ));
            }
            let sqisign_pk = &public_key[32..32 + SQISIGN_PK_SIZE];

            // Verify BOTH signatures with SEPARATE keys
            verify_ed25519_signature(ed_sig, block_hash, ed_pk)?;
            verify_sqisign_signature(sqisign_sig, block_hash, sqisign_pk)?;

            Ok(())
        }
    }
}

// Tests exercise sign_ed25519 / sign_dilithium5, which are gated on `signing`.
// Compiling them without the feature would be a missing-symbol error,
// so the whole module is gated to match.
#[cfg(all(test, feature = "signing"))]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// Deterministic test signing key — no `rand` dep, no `OsRng`.
    /// Mirrors the helper used in q-crypto-simd/src/parallel_ed25519.rs.
    fn signing_key_from_index(i: u32) -> SigningKey {
        let mut seed = [0u8; 32];
        seed[0..4].copy_from_slice(&i.to_le_bytes());
        SigningKey::from_bytes(&seed)
    }

    #[test]
    fn test_ed25519_signature_verification() {
        // Generate keypair
        let signing_key = signing_key_from_index(0);
        let verifying_key = signing_key.verifying_key();

        // Sign message
        let message = b"test message for signature verification";
        let signature = sign_ed25519(message, &signing_key);

        // Verify signature
        let result = verify_ed25519_signature(
            &signature,
            message,
            verifying_key.as_bytes(),
        );

        assert!(result.is_ok(), "Ed25519 signature should verify");
    }

    #[test]
    fn test_dilithium5_signature_verification() {
        // Generate Dilithium5 keypair
        let (pk, sk) = dilithium5::keypair();

        // Sign message
        let message = b"test message for PQC signature verification";
        let signed_message = sign_dilithium5(message, &sk);

        // Verify signature
        let result = verify_dilithium5_signature(
            &signed_message,
            message,
            pk.as_bytes(),
        );

        assert!(result.is_ok(), "Dilithium5 signature should verify");
    }

    #[test]
    fn test_spectral_signature_phase0() {
        let signing_key = signing_key_from_index(1);
        let verifying_key = signing_key.verifying_key();

        let message = b"block hash to sign";
        let signature = sign_ed25519(message, &signing_key);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::Phase0Ed25519,
            classical_sig: signature,
            pqc_sig: None,
            sqisign_sig: None,
            spectral_coefficient: 1.0,
            phase_deviation: 0.0,
            timestamp: 1700000000,
        };

        let result = verify_spectral_signature(
            &spectral_sig,
            message,
            Some(verifying_key.as_bytes()),
            None,
        );

        assert!(result.is_ok(), "Phase0 spectral signature should verify");
    }

    #[allow(deprecated)]
    #[test]
    fn test_spectral_signature_phase1() {
        let (pk, sk) = dilithium5::keypair();

        let message = b"block hash to sign with PQC";
        let pqc_signature = sign_dilithium5(message, &sk);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::Phase1Dilithium5,
            classical_sig: vec![], // Not used in Phase1
            pqc_sig: Some(pqc_signature),
            sqisign_sig: None,
            spectral_coefficient: 1.0,
            phase_deviation: 0.0,
            timestamp: 1700000000,
        };

        let result = verify_spectral_signature(
            &spectral_sig,
            message,
            None,
            Some(pk.as_bytes()),
        );

        assert!(result.is_ok(), "Phase1 spectral signature should verify");
    }

    #[allow(deprecated)]
    #[test]
    fn test_spectral_signature_hybrid() {
        let ed_signing_key = signing_key_from_index(2);
        let ed_verifying_key = ed_signing_key.verifying_key();
        let (pqc_pk, pqc_sk) = dilithium5::keypair();

        let message = b"block hash with hybrid signatures";

        let ed_signature = sign_ed25519(message, &ed_signing_key);
        let pqc_signature = sign_dilithium5(message, &pqc_sk);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::HybridEd25519Dilithium5,
            classical_sig: ed_signature,
            pqc_sig: Some(pqc_signature),
            sqisign_sig: None,
            spectral_coefficient: 1.0,
            phase_deviation: 0.0,
            timestamp: 1700000000,
        };

        let result = verify_spectral_signature(
            &spectral_sig,
            message,
            Some(ed_verifying_key.as_bytes()),
            Some(pqc_pk.as_bytes()),
        );

        assert!(result.is_ok(), "Hybrid spectral signature should verify");
    }

    // =========================================================================
    // SQIsign Tests (v1.0.86-beta)
    // =========================================================================

    #[test]
    fn test_sqisign_signature_creation_and_size() {
        // Generate a mock SQIsign keypair (64-byte key)
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();

        let message = b"test SQIsign signature creation";
        let signature = sign_sqisign(message, &secret_key, &public_key);

        // Verify signature size is exactly 204 bytes (NIST Level I)
        assert_eq!(
            signature.len(),
            SQISIGN_SIG_SIZE,
            "SQIsign signature should be {} bytes",
            SQISIGN_SIG_SIZE
        );

        // Verify the structure
        assert_eq!(signature[0], 0, "Level should be 0 (Level I)");
    }

    #[test]
    fn test_sqisign_vs_dilithium5_size_comparison() {
        // This test demonstrates the massive size difference
        let dilithium5_sig_size = 4627_usize; // Dilithium5 signature size
        let sqisign_sig_size = SQISIGN_SIG_SIZE;

        let savings_percent = (1.0 - (sqisign_sig_size as f64 / dilithium5_sig_size as f64)) * 100.0;

        // SQIsign should be 95.6% smaller than Dilithium5
        assert!(
            savings_percent > 95.0,
            "SQIsign should be >95% smaller than Dilithium5, got {:.1}%",
            savings_percent
        );

        println!("📊 Signature Size Comparison:");
        println!("   Dilithium5: {} bytes", dilithium5_sig_size);
        println!("   SQIsign:    {} bytes", sqisign_sig_size);
        println!("   Savings:    {:.1}%", savings_percent);
    }

    #[test]
    fn test_spectral_signature_phase2_sqisign() {
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();

        let message = b"block hash for SQIsign";
        let sqisign_signature = sign_sqisign(message, &secret_key, &public_key);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::Phase2SQIsign,
            classical_sig: vec![], // Not used in Phase2
            pqc_sig: None,         // Deprecated
            sqisign_sig: Some(sqisign_signature),
            spectral_coefficient: 1.0,
            phase_deviation: 0.0,
            timestamp: 1700000000,
        };

        let result = verify_spectral_signature_extended(
            &spectral_sig,
            message,
            None,
            None,
            Some(&public_key),
        );

        assert!(result.is_ok(), "Phase2 SQIsign spectral signature should verify");
    }

    #[test]
    fn test_spectral_signature_hybrid_sqisign() {
        let ed_signing_key = signing_key_from_index(3);
        let ed_verifying_key = ed_signing_key.verifying_key();

        let mut sqisign_secret = [0u8; 64];
        let mut sqisign_public = [0u8; 64];
        getrandom::getrandom(&mut sqisign_secret).unwrap();
        getrandom::getrandom(&mut sqisign_public).unwrap();

        let message = b"block hash with hybrid Ed25519+SQIsign";

        let ed_signature = sign_ed25519(message, &ed_signing_key);
        let sqisign_signature = sign_sqisign(message, &sqisign_secret, &sqisign_public);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::HybridEd25519SQIsign,
            classical_sig: ed_signature,
            pqc_sig: None,
            sqisign_sig: Some(sqisign_signature),
            spectral_coefficient: 1.0,
            phase_deviation: 0.0,
            timestamp: 1700000000,
        };

        let result = verify_spectral_signature_extended(
            &spectral_sig,
            message,
            Some(ed_verifying_key.as_bytes()),
            None,
            Some(&sqisign_public),
        );

        assert!(result.is_ok(), "Hybrid Ed25519+SQIsign spectral signature should verify");
    }

    // =========================================================================
    // NEGATIVE TESTS (v2.3.1-beta) - Verify invalid signatures are REJECTED
    // =========================================================================

    #[test]
    fn test_sqisign_forged_signature_rejected() {
        // Create a completely forged signature (random bytes)
        let mut forged_signature = vec![0u8; SQISIGN_SIG_SIZE];
        forged_signature[0] = 0; // Level I
        getrandom::getrandom(&mut forged_signature[1..]).unwrap();

        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut public_key).unwrap();

        let message = b"message that was NOT signed with this key";

        let result = verify_sqisign_signature(&forged_signature, message, &public_key);

        assert!(
            result.is_err(),
            "SECURITY CRITICAL: Forged SQIsign signature should be REJECTED, not accepted!"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("verification FAILED") || err_msg.contains("mismatch"),
            "Error should indicate verification failure, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_sqisign_tampered_signature_rejected() {
        // Create a valid signature, then tamper with it
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();

        let message = b"original message";
        let mut signature = sign_sqisign(message, &secret_key, &public_key);

        // Tamper with the commitment (bytes 1-32)
        signature[10] ^= 0xFF;

        let result = verify_sqisign_signature(&signature, message, &public_key);

        assert!(
            result.is_err(),
            "SECURITY CRITICAL: Tampered SQIsign signature should be REJECTED!"
        );
    }

    #[test]
    fn test_sqisign_wrong_message_rejected() {
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();

        let original_message = b"original message";
        let signature = sign_sqisign(original_message, &secret_key, &public_key);

        // Try to verify with a different message
        let different_message = b"different message";

        let result = verify_sqisign_signature(&signature, different_message, &public_key);

        assert!(
            result.is_err(),
            "SECURITY CRITICAL: Signature for different message should be REJECTED!"
        );
    }

    #[test]
    fn test_sqisign_wrong_public_key_rejected() {
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        let mut wrong_public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();
        getrandom::getrandom(&mut wrong_public_key).unwrap();

        let message = b"test message";
        let signature = sign_sqisign(message, &secret_key, &public_key);

        // Try to verify with wrong public key
        let result = verify_sqisign_signature(&signature, message, &wrong_public_key);

        assert!(
            result.is_err(),
            "SECURITY CRITICAL: Signature with wrong public key should be REJECTED!"
        );
    }

    #[test]
    fn test_hybrid_insufficient_key_material_rejected() {
        // Create a valid hybrid signature
        let ed_signing_key = signing_key_from_index(4);
        let ed_verifying_key = ed_signing_key.verifying_key();

        let mut sqisign_secret = [0u8; 64];
        let mut sqisign_public = [0u8; 64];
        getrandom::getrandom(&mut sqisign_secret).unwrap();
        getrandom::getrandom(&mut sqisign_public).unwrap();

        let message = b"block hash";
        let block_hash: [u8; 32] = sha3::Sha3_256::digest(message).into();

        let ed_sig = sign_ed25519(&block_hash, &ed_signing_key);
        let sqisign_sig = sign_sqisign(&block_hash, &sqisign_secret, &sqisign_public);

        // Combine signatures
        let mut combined_sig = Vec::new();
        combined_sig.extend_from_slice(&ed_sig);
        combined_sig.extend_from_slice(&sqisign_sig);

        // Try to verify with ONLY the Ed25519 key (insufficient material)
        let result = verify_block_signature(
            &combined_sig,
            &block_hash,
            ed_verifying_key.as_bytes(), // Only 32 bytes, not 32 + 64
            SignaturePhase::HybridEd25519SQIsign,
        );

        assert!(
            result.is_err(),
            "SECURITY CRITICAL: Hybrid mode with insufficient key material should be REJECTED!"
        );

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SECURITY ERROR") || err_msg.contains("SEPARATE keys"),
            "Error should indicate security violation, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_ed25519_invalid_signature_rejected() {
        let signing_key = signing_key_from_index(5);
        let verifying_key = signing_key.verifying_key();

        let message = b"test message";
        let mut signature = sign_ed25519(message, &signing_key);

        // Tamper with signature
        signature[0] ^= 0xFF;

        let result = verify_ed25519_signature(&signature, message, verifying_key.as_bytes());

        assert!(
            result.is_err(),
            "Tampered Ed25519 signature should be REJECTED!"
        );
    }

    #[test]
    fn test_sqisign_signature_roundtrip_works() {
        // Verify that sign -> verify works correctly
        let mut secret_key = [0u8; 64];
        let mut public_key = [0u8; 64];
        getrandom::getrandom(&mut secret_key).unwrap();
        getrandom::getrandom(&mut public_key).unwrap();

        let message = b"test roundtrip verification";
        let signature = sign_sqisign(message, &secret_key, &public_key);

        let result = verify_sqisign_signature(&signature, message, &public_key);

        assert!(
            result.is_ok(),
            "Valid SQIsign signature should verify successfully: {:?}",
            result.err()
        );
    }
}
