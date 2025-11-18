/// Post-Quantum Signature Verification Module
/// v1.0.15-beta: Implements Dilithium5 signature verification for blocks
///
/// This module provides crypto-agile signature verification supporting:
/// - Phase 0: Ed25519 classical signatures
/// - Phase 1: Dilithium5 post-quantum signatures
/// - Hybrid: Dual Ed25519 + Dilithium5 for transition

use crate::block::{SignaturePhase, SpectralSignature};
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SignedMessage};

/// Verify a spectral signature based on its crypto phase
pub fn verify_spectral_signature(
    signature: &SpectralSignature,
    message: &[u8],
    public_key_ed25519: Option<&[u8]>,
    public_key_dilithium5: Option<&[u8]>,
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
    }
}

/// Verify Ed25519 signature (Phase 0)
fn verify_ed25519_signature(signature: &[u8], message: &[u8], public_key: &[u8]) -> Result<()> {
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
fn verify_dilithium5_signature(
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

/// Create a Dilithium5 signature (for Phase 1)
#[cfg(feature = "signing")]
pub fn sign_dilithium5(message: &[u8], secret_key: &dilithium5::SecretKey) -> Vec<u8> {
    dilithium5::sign(message, secret_key).as_bytes().to_vec()
}

/// Verify a block hash signature (used for block finality certificates)
pub fn verify_block_signature(
    signature: &[u8],
    block_hash: &[u8; 32],
    public_key: &[u8],
    phase: SignaturePhase,
) -> Result<()> {
    match phase {
        SignaturePhase::Phase0Ed25519 => verify_ed25519_signature(signature, block_hash, public_key),

        SignaturePhase::Phase1Dilithium5 => {
            verify_dilithium5_signature(signature, block_hash, public_key)
        }

        SignaturePhase::HybridEd25519Dilithium5 => {
            // For hybrid mode on block signatures, we need to split the signature
            // Format: [ed25519_sig (64 bytes)] || [dilithium5_sig]
            if signature.len() < 64 {
                return Err(anyhow!("Hybrid signature too short"));
            }

            let (ed_sig, pqc_sig) = signature.split_at(64);

            verify_ed25519_signature(ed_sig, block_hash, public_key)?;
            verify_dilithium5_signature(pqc_sig, block_hash, public_key)?;

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn test_ed25519_signature_verification() {
        // Generate keypair
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
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
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let message = b"block hash to sign";
        let signature = sign_ed25519(message, &signing_key);

        let spectral_sig = SpectralSignature {
            validator: [0u8; 32],
            crypto_phase: SignaturePhase::Phase0Ed25519,
            classical_sig: signature,
            pqc_sig: None,
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

    #[test]
    fn test_spectral_signature_hybrid() {
        let ed_signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
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
}
