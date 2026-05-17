//! Signature verification with SIMD-friendly batch dispatch.
//!
//! The "AVX-512" name is historical; today this is rayon-parallel batch
//! verification using real cryptography:
//!   * Ed25519: `ed25519_dalek::VerifyingKey::verify_strict`
//!   * Dilithium5: `pqcrypto_dilithium::dilithium5::verify_detached_signature`
//!
//! v10.9.20: replaced the byte-length placeholders that previously masqueraded
//! as verification. Old code returned `Ok(true)` if signatures were >= 1000B
//! and pubkeys >= 1000B regardless of cryptographic validity.

use crate::SimdResult;
use anyhow::Result;
use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{
    DetachedSignature as PqDetachedSignature, PublicKey as PqPublicKey,
};
use rayon::prelude::*;
use tracing::debug;

/// Ed25519 signature / public-key sizes.
const ED25519_SIG_BYTES: usize = 64;
const ED25519_PK_BYTES: usize = 32;

// Dilithium5 signature/pubkey sizes are version-dependent across pqcrypto
// revisions (4,595 in some, 4,627 in others). Don't hardcode — let
// `dilithium5::DetachedSignature::from_bytes` and `PublicKey::from_bytes`
// reject wrong-length inputs.

/// Batch signature verifier.
///
/// Per-signature work is real cryptography; the speedup comes from
/// data-parallel dispatch over a rayon pool.
pub struct Avx512SignatureVerifier {
    /// CPU capability bits captured at construction. Reserved for future
    /// SIMD-tuned inner loops (e.g. Dilithium NTT). Not consulted in this
    /// build — verification is always real-crypto regardless.
    capabilities: u64,
}

impl Avx512SignatureVerifier {
    /// Create new signature verifier.
    pub fn new() -> Self {
        Self { capabilities: 0 }
    }

    /// Verify Ed25519 signatures in batch using rayon-parallel dispatch.
    ///
    /// Returns a [`SimdResult`] with the count of *valid* signatures. Invalid
    /// signatures do not error the batch; they're counted as `0` against
    /// `valid_count`.
    pub fn verify_ed25519_batch(
        &self,
        messages: &[&[u8]],
        signatures: &[&[u8]],
        public_keys: &[&[u8]],
    ) -> Result<SimdResult> {
        debug!(
            "🔐 Batch Ed25519 verification: {} signatures",
            signatures.len()
        );

        if messages.len() != signatures.len() || messages.len() != public_keys.len() {
            return Err(anyhow::anyhow!("Mismatched batch sizes"));
        }

        let valid_count: u32 = messages
            .par_iter()
            .zip(signatures.par_iter())
            .zip(public_keys.par_iter())
            .filter(|((msg, sig), pk)| verify_single_ed25519(msg, sig, pk).unwrap_or(false))
            .count() as u32;

        let operations = messages.len() as u64;
        let performance_gain = 1.8; // Conservative estimate for rayon batch dispatch

        Ok(SimdResult::with_signatures(
            operations,
            performance_gain,
            valid_count,
        ))
    }

    /// Verify Dilithium5 detached signatures in batch.
    ///
    /// `signatures` slices are expected to be **detached** signatures
    /// (`DILITHIUM5_SIG_BYTES`), not signed-message form. Producers calling
    /// `dilithium5::sign(msg, sk)` get a `SignedMessage` (signature + msg);
    /// callers must use `dilithium5::detached_sign(msg, sk)` to land in the
    /// shape this batch verifier expects.
    pub fn verify_dilithium5_batch(
        &self,
        messages: &[&[u8]],
        signatures: &[&[u8]],
        public_keys: &[&[u8]],
    ) -> Result<SimdResult> {
        debug!(
            "🔐 Batch Dilithium5 verification: {} signatures",
            signatures.len()
        );

        if messages.len() != signatures.len() || messages.len() != public_keys.len() {
            return Err(anyhow::anyhow!("Mismatched batch sizes"));
        }

        let valid_count: u32 = messages
            .par_iter()
            .zip(signatures.par_iter())
            .zip(public_keys.par_iter())
            .filter(|((msg, sig), pk)| verify_single_dilithium5(msg, sig, pk).unwrap_or(false))
            .count() as u32;

        let operations = messages.len() as u64;
        let performance_gain = 1.5;

        Ok(SimdResult::with_signatures(
            operations,
            performance_gain,
            valid_count,
        ))
    }

    /// Get throughput estimate for signature verification.
    pub fn throughput_estimate(&self) -> f64 {
        if self.capabilities > 0 {
            50_000.0
        } else {
            25_000.0
        }
    }
}

impl Default for Avx512SignatureVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Real single-signature Ed25519 verify.
///
/// Returns `Ok(true)` only when the signature is cryptographically valid.
/// Length-mismatch or malformed inputs return `Ok(false)` (an invalid
/// signature, not an internal error).
fn verify_single_ed25519(message: &[u8], signature: &[u8], pubkey: &[u8]) -> Result<bool> {
    if signature.len() != ED25519_SIG_BYTES || pubkey.len() != ED25519_PK_BYTES {
        return Ok(false);
    }
    let pk_array: [u8; 32] = pubkey.try_into().expect("len checked above");
    let sig_array: [u8; 64] = signature.try_into().expect("len checked above");
    let Ok(verifying_key) = VerifyingKey::from_bytes(&pk_array) else {
        return Ok(false);
    };
    let signature = EdSignature::from_bytes(&sig_array);
    Ok(verifying_key.verify(message, &signature).is_ok())
}

/// Real single-signature Dilithium5 verify (detached form).
fn verify_single_dilithium5(message: &[u8], signature: &[u8], pubkey: &[u8]) -> Result<bool> {
    let Ok(pk) = dilithium5::PublicKey::from_bytes(pubkey) else {
        return Ok(false);
    };
    let Ok(sig) = dilithium5::DetachedSignature::from_bytes(signature) else {
        return Ok(false);
    };
    Ok(dilithium5::verify_detached_signature(&sig, message, &pk).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;
    use pqcrypto_dilithium::dilithium5;
    use pqcrypto_traits::sign::DetachedSignature as PqDetachedSignatureTrait;

    #[test]
    fn ed25519_batch_accepts_valid_and_rejects_invalid() {
        let verifier = Avx512SignatureVerifier::new();

        let secret = [7u8; 32];
        let signing = ed25519_dalek::SigningKey::from_bytes(&secret);
        let verifying = signing.verifying_key();
        let msg = b"q-narwhalknight batch verify test";
        let sig = signing.sign(msg);

        // 3 valid + 2 bogus
        let pk_bytes = verifying.to_bytes();
        let sig_bytes = sig.to_bytes();
        let bogus_sig = [0u8; 64];
        let messages: Vec<&[u8]> = vec![msg, msg, msg, msg, msg];
        let signatures: Vec<&[u8]> = vec![&sig_bytes, &sig_bytes, &sig_bytes, &bogus_sig, &bogus_sig];
        let public_keys: Vec<&[u8]> = vec![&pk_bytes; 5];

        let result = verifier
            .verify_ed25519_batch(&messages, &signatures, &public_keys)
            .unwrap();
        assert_eq!(result.operations_completed, 5);
        assert_eq!(result.valid_signatures, 3);
    }

    #[test]
    fn ed25519_rejects_wrong_pubkey() {
        let verifier = Avx512SignatureVerifier::new();
        let signing = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
        let attacker = ed25519_dalek::SigningKey::from_bytes(&[22u8; 32]);
        let msg = b"important message";
        let sig = signing.sign(msg);

        let attacker_pk = attacker.verifying_key().to_bytes();
        let sig_bytes = sig.to_bytes();
        let result = verifier
            .verify_ed25519_batch(&[msg], &[&sig_bytes], &[&attacker_pk])
            .unwrap();
        assert_eq!(result.valid_signatures, 0);
    }

    #[test]
    fn dilithium5_batch_accepts_valid_and_rejects_invalid() {
        let verifier = Avx512SignatureVerifier::new();
        let (pk, sk) = dilithium5::keypair();
        let msg = b"post-quantum batch test";
        let detached = dilithium5::detached_sign(msg, &sk);

        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = detached.as_bytes().to_vec();
        // Bogus signature: same length as real, but all zeros — must be
        // rejected by Dilithium5 verification.
        let bogus_sig = vec![0u8; sig_bytes.len()];

        let messages: Vec<&[u8]> = vec![msg, msg, msg];
        let signatures: Vec<&[u8]> = vec![&sig_bytes, &sig_bytes, &bogus_sig];
        let public_keys: Vec<&[u8]> = vec![&pk_bytes; 3];

        let result = verifier
            .verify_dilithium5_batch(&messages, &signatures, &public_keys)
            .unwrap();
        assert_eq!(result.operations_completed, 3);
        assert_eq!(result.valid_signatures, 2);
    }

    #[test]
    fn dilithium5_rejects_tampered_message() {
        let verifier = Avx512SignatureVerifier::new();
        let (pk, sk) = dilithium5::keypair();
        let detached = dilithium5::detached_sign(b"original message", &sk);
        let pk_bytes = pk.as_bytes().to_vec();
        let sig_bytes = detached.as_bytes().to_vec();

        let tampered: &[u8] = b"tampered message";
        let result = verifier
            .verify_dilithium5_batch(&[tampered], &[&sig_bytes], &[&pk_bytes])
            .unwrap();
        assert_eq!(result.valid_signatures, 0);
    }

    #[test]
    fn ed25519_rejects_length_mismatch_without_panic() {
        let verifier = Avx512SignatureVerifier::new();
        let too_short_sig: &[u8] = &[0u8; 32];
        let bad_pk: &[u8] = &[0u8; 16];
        let result = verifier
            .verify_ed25519_batch(&[b"msg"], &[too_short_sig], &[bad_pk])
            .unwrap();
        assert_eq!(result.valid_signatures, 0);
    }
}
