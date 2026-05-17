//! Trustee key management
//!
//! Handles generation and storage of trustee keys (ML-KEM + Dilithium).

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};
use pqcrypto_traits::sign::DetachedSignature as DetachedSignatureTrait;

use crate::crypto::kem;
use crate::error::{TemporalError, TemporalResult};

/// Public key for a trustee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrusteePublicKey {
    /// Unique identifier (hash of public keys)
    pub id: [u8; 32],

    /// ML-KEM-1024 public key for key encapsulation
    pub kem_public_key: Vec<u8>,

    /// Dilithium public key for signatures (optional)
    pub signature_public_key: Option<Vec<u8>>,

    /// Human-readable name (optional)
    pub name: Option<String>,

    /// Metadata
    pub created_at: u64,
}

/// Private key for a trustee (sensitive!)
#[derive(Debug, Zeroize, ZeroizeOnDrop)]
pub struct TrusteePrivateKey {
    /// Unique identifier
    #[zeroize(skip)]
    pub id: [u8; 32],

    /// ML-KEM-1024 secret key
    pub kem_secret_key: Vec<u8>,

    /// Dilithium secret key for signatures (optional)
    pub signature_secret_key: Option<Vec<u8>>,
}

/// Complete key pair for a trustee
pub struct TrusteeKeyPair {
    pub public_key: TrusteePublicKey,
    pub private_key: TrusteePrivateKey,
}

impl TrusteePublicKey {
    /// Generate a new trustee key pair
    pub fn generate(name: Option<String>) -> TemporalResult<TrusteeKeyPair> {
        // Generate ML-KEM key pair
        let (kem_pk, kem_sk) = kem::MlKem1024::generate_keypair();

        // Generate signature key pair (optional, using Dilithium)
        let (sig_pk, sig_sk) = generate_dilithium_keypair()?;

        // Compute trustee ID from public keys
        let id = compute_trustee_id(&kem_pk, sig_pk.as_deref());

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let public_key = TrusteePublicKey {
            id,
            kem_public_key: kem_pk,
            signature_public_key: sig_pk,
            name,
            created_at: timestamp,
        };

        let private_key = TrusteePrivateKey {
            id,
            kem_secret_key: kem_sk,
            signature_secret_key: sig_sk,
        };

        Ok(TrusteeKeyPair {
            public_key,
            private_key,
        })
    }

    /// Get the size of the public key in bytes
    pub fn size(&self) -> usize {
        32 + self.kem_public_key.len()
            + self.signature_public_key.as_ref().map(|k| k.len()).unwrap_or(0)
            + self.name.as_ref().map(|n| n.len()).unwrap_or(0)
            + 8
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> TemporalResult<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| TemporalError::SerializationFailed(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> TemporalResult<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| TemporalError::DeserializationFailed(e.to_string()))
    }
}

impl TrusteePrivateKey {
    /// Decrypt a share encrypted for this trustee
    pub fn decrypt_share(
        &self,
        encrypted_share: &crate::crypto::EncryptedShare,
    ) -> TemporalResult<Vec<u8>> {
        // Verify this share is for us
        if encrypted_share.trustee_id != self.id {
            return Err(TemporalError::ShareDecryptionFailed {
                trustee_id: hex::encode(encrypted_share.trustee_id),
            });
        }

        kem::decrypt_share(encrypted_share, &self.kem_secret_key)
    }
}

/// Compute trustee ID from public keys
fn compute_trustee_id(kem_pk: &[u8], sig_pk: Option<&[u8]>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"TemporalShield-TrusteeID-v2");
    hasher.update(kem_pk);
    if let Some(pk) = sig_pk {
        hasher.update(pk);
    }
    *hasher.finalize().as_bytes()
}

/// Generate Dilithium signature key pair
fn generate_dilithium_keypair() -> TemporalResult<(Option<Vec<u8>>, Option<Vec<u8>>)> {
    use pqcrypto_dilithium::dilithium5;
    use pqcrypto_traits::sign::{PublicKey, SecretKey};

    let (pk, sk) = dilithium5::keypair();
    Ok((
        Some(pk.as_bytes().to_vec()),
        Some(sk.as_bytes().to_vec()),
    ))
}

/// Sign data with trustee's signature key
pub fn sign_data(
    private_key: &TrusteePrivateKey,
    data: &[u8],
) -> TemporalResult<Vec<u8>> {
    use pqcrypto_dilithium::dilithium5;
    use pqcrypto_traits::sign::SecretKey;

    let sk_bytes = private_key.signature_secret_key.as_ref()
        .ok_or(TemporalError::SignatureFailed("No signature key".to_string()))?;

    let sk = dilithium5::SecretKey::from_bytes(sk_bytes)
        .map_err(|_| TemporalError::SignatureFailed("Invalid secret key".to_string()))?;

    let signature = dilithium5::detached_sign(data, &sk);

    Ok(signature.as_bytes().to_vec())
}

/// Verify a signature
pub fn verify_signature(
    public_key: &TrusteePublicKey,
    data: &[u8],
    signature: &[u8],
) -> TemporalResult<bool> {
    use pqcrypto_dilithium::dilithium5;
    use pqcrypto_traits::sign::PublicKey;

    let pk_bytes = public_key.signature_public_key.as_ref()
        .ok_or(TemporalError::SignatureVerificationFailed)?;

    let pk = dilithium5::PublicKey::from_bytes(pk_bytes)
        .map_err(|_| TemporalError::SignatureVerificationFailed)?;

    let sig = dilithium5::DetachedSignature::from_bytes(signature)
        .map_err(|_| TemporalError::SignatureVerificationFailed)?;

    match dilithium5::verify_detached_signature(&sig, data, &pk) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Generate multiple trustee key pairs
pub fn generate_trustees(count: usize) -> TemporalResult<Vec<TrusteeKeyPair>> {
    (0..count)
        .map(|i| TrusteePublicKey::generate(Some(format!("Trustee-{}", i + 1))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_trustee() {
        let keypair = TrusteePublicKey::generate(Some("Test Trustee".to_string())).unwrap();

        assert_ne!(keypair.public_key.id, [0u8; 32]);
        assert!(!keypair.public_key.kem_public_key.is_empty());
        assert!(keypair.public_key.signature_public_key.is_some());
    }

    #[test]
    fn test_trustee_serialization() {
        let keypair = TrusteePublicKey::generate(None).unwrap();

        let bytes = keypair.public_key.to_bytes().unwrap();
        let restored = TrusteePublicKey::from_bytes(&bytes).unwrap();

        assert_eq!(keypair.public_key.id, restored.id);
        assert_eq!(keypair.public_key.kem_public_key, restored.kem_public_key);
    }

    #[test]
    fn test_sign_verify() {
        let keypair = TrusteePublicKey::generate(None).unwrap();
        let data = b"Test data to sign";

        let signature = sign_data(&keypair.private_key, data).unwrap();
        let valid = verify_signature(&keypair.public_key, data, &signature).unwrap();

        assert!(valid);
    }

    #[test]
    fn test_sign_verify_wrong_data() {
        let keypair = TrusteePublicKey::generate(None).unwrap();
        let data = b"Test data to sign";
        let wrong_data = b"Wrong data";

        let signature = sign_data(&keypair.private_key, data).unwrap();
        let valid = verify_signature(&keypair.public_key, wrong_data, &signature).unwrap();

        assert!(!valid);
    }

    #[test]
    fn test_generate_multiple_trustees() {
        let trustees = generate_trustees(5).unwrap();
        assert_eq!(trustees.len(), 5);

        // All should have unique IDs
        let ids: std::collections::HashSet<_> = trustees.iter()
            .map(|t| t.public_key.id)
            .collect();
        assert_eq!(ids.len(), 5);
    }
}
