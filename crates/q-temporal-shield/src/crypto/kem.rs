//! Key Encapsulation Mechanism (KEM) for TemporalShield
//!
//! Uses ML-KEM (Kyber) for post-quantum secure key exchange.

use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{PublicKey, SecretKey, SharedSecret, Ciphertext};
use crate::error::{TemporalError, TemporalResult};

/// ML-KEM-1024 (Kyber-1024) key encapsulation
pub struct MlKem1024;

/// Public key for ML-KEM-1024
pub type KemPublicKey = kyber1024::PublicKey;

/// Secret key for ML-KEM-1024
pub type KemSecretKey = kyber1024::SecretKey;

/// Ciphertext from encapsulation
pub type KemCiphertext = kyber1024::Ciphertext;

impl MlKem1024 {
    /// Generate a new key pair
    pub fn generate_keypair() -> (Vec<u8>, Vec<u8>) {
        let (pk, sk) = kyber1024::keypair();
        (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
    }

    /// Encapsulate: Generate a shared secret and ciphertext
    pub fn encapsulate(public_key: &[u8]) -> TemporalResult<(Vec<u8>, [u8; 32])> {
        let pk = kyber1024::PublicKey::from_bytes(public_key)
            .map_err(|_| TemporalError::KemEncapsulationFailed("Invalid public key".to_string()))?;

        let (shared_secret, ciphertext) = kyber1024::encapsulate(&pk);

        // Convert shared secret to 32 bytes
        let mut ss_bytes = [0u8; 32];
        ss_bytes.copy_from_slice(shared_secret.as_bytes());

        Ok((ciphertext.as_bytes().to_vec(), ss_bytes))
    }

    /// Decapsulate: Recover the shared secret from ciphertext
    pub fn decapsulate(ciphertext: &[u8], secret_key: &[u8]) -> TemporalResult<[u8; 32]> {
        let ct = kyber1024::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| TemporalError::KemDecapsulationFailed("Invalid ciphertext".to_string()))?;

        let sk = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| TemporalError::KemDecapsulationFailed("Invalid secret key".to_string()))?;

        let shared_secret = kyber1024::decapsulate(&ct, &sk);

        let mut ss_bytes = [0u8; 32];
        ss_bytes.copy_from_slice(shared_secret.as_bytes());

        Ok(ss_bytes)
    }

    /// Get the public key size in bytes
    pub fn public_key_size() -> usize {
        kyber1024::public_key_bytes()
    }

    /// Get the secret key size in bytes
    pub fn secret_key_size() -> usize {
        kyber1024::secret_key_bytes()
    }

    /// Get the ciphertext size in bytes
    pub fn ciphertext_size() -> usize {
        kyber1024::ciphertext_bytes()
    }

    /// Get the shared secret size in bytes
    pub fn shared_secret_size() -> usize {
        32 // Always 32 bytes
    }
}

/// Encrypt a share for a trustee using hybrid KEM+AEAD
pub fn encrypt_share_for_trustee(
    share_data: &[u8],
    trustee_public_key: &[u8],
    trustee_id: [u8; 32],
    index: u32,
) -> TemporalResult<super::EncryptedShare> {
    // 1. Encapsulate to get shared secret
    let (kem_ciphertext, shared_secret) = MlKem1024::encapsulate(trustee_public_key)?;

    // 2. Derive AEAD key from shared secret
    let aead_key = super::hash::derive_key_material("TemporalShield-AEAD-v2", &shared_secret);

    // 3. Generate random nonce
    let mut nonce = [0u8; 24];
    super::rand::fill_random(&mut nonce)?;

    // 4. Encrypt share with AEAD
    let encrypted_data = super::aead::encrypt(&aead_key, &nonce, share_data)?;

    Ok(super::EncryptedShare {
        trustee_id,
        index,
        encrypted_data,
        kem_ciphertext,
        nonce,
    })
}

/// Decrypt a share using trustee's secret key
pub fn decrypt_share(
    encrypted_share: &super::EncryptedShare,
    trustee_secret_key: &[u8],
) -> TemporalResult<Vec<u8>> {
    // 1. Decapsulate to recover shared secret
    let shared_secret = MlKem1024::decapsulate(&encrypted_share.kem_ciphertext, trustee_secret_key)?;

    // 2. Derive AEAD key
    let aead_key = super::hash::derive_key_material("TemporalShield-AEAD-v2", &shared_secret);

    // 3. Decrypt share
    let share_data = super::aead::decrypt(&aead_key, &encrypted_share.nonce, &encrypted_share.encrypted_data)?;

    Ok(share_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let (pk, sk) = MlKem1024::generate_keypair();
        assert_eq!(pk.len(), MlKem1024::public_key_size());
        assert_eq!(sk.len(), MlKem1024::secret_key_size());
    }

    #[test]
    fn test_encapsulate_decapsulate() {
        let (pk, sk) = MlKem1024::generate_keypair();

        let (ciphertext, shared_secret1) = MlKem1024::encapsulate(&pk).unwrap();
        let shared_secret2 = MlKem1024::decapsulate(&ciphertext, &sk).unwrap();

        assert_eq!(shared_secret1, shared_secret2);
    }

    #[test]
    fn test_encrypt_decrypt_share() {
        let (pk, sk) = MlKem1024::generate_keypair();
        let trustee_id = super::super::hash::blake3_hash(&pk);

        let share_data = b"This is a secret share";
        let encrypted = encrypt_share_for_trustee(share_data, &pk, trustee_id, 1).unwrap();

        let decrypted = decrypt_share(&encrypted, &sk).unwrap();
        assert_eq!(decrypted, share_data);
    }
}
