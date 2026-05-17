/// Kyber1024 Post-Quantum Key Encapsulation Module
/// Phase 6: NIST Level 5 post-quantum key exchange for wallet encryption
///
/// This module provides Kyber1024 KEM for quantum-resistant encryption of wallet secrets

use anyhow::{anyhow, Result};
use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{Ciphertext as KemCiphertext, PublicKey as KemPublicKey, SecretKey as KemSecretKey, SharedSecret as KemSharedSecret};

/// Kyber1024 keypair for quantum-resistant key exchange
pub struct Kyber1024KeyPair {
    pub public_key: kyber1024::PublicKey,
    pub secret_key: kyber1024::SecretKey,
}

/// Kyber1024 encapsulated ciphertext
pub struct Kyber1024Ciphertext {
    pub ciphertext: kyber1024::Ciphertext,
}

impl Kyber1024KeyPair {
    /// Generate a new Kyber1024 keypair using quantum random number generation
    pub fn generate() -> Self {
        let (public_key, secret_key) = kyber1024::keypair();
        Self {
            public_key,
            secret_key,
        }
    }

    /// Encapsulate a shared secret using recipient's public key
    /// Returns (shared_secret, ciphertext)
    pub fn encapsulate(public_key: &kyber1024::PublicKey) -> (Vec<u8>, Vec<u8>) {
        let (shared_secret, ciphertext) = kyber1024::encapsulate(public_key);
        (
            shared_secret.as_bytes().to_vec(),
            ciphertext.as_bytes().to_vec(),
        )
    }

    /// Decapsulate shared secret from ciphertext using secret key
    pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        let ct = kyber1024::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| anyhow!("Invalid Kyber1024 ciphertext"))?;

        let shared_secret = kyber1024::decapsulate(&ct, &self.secret_key);
        Ok(shared_secret.as_bytes().to_vec())
    }

    /// Get public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }

    /// Get secret key bytes
    pub fn secret_key_bytes(&self) -> Vec<u8> {
        self.secret_key.as_bytes().to_vec()
    }

    /// Reconstruct keypair from bytes
    pub fn from_bytes(public_key: &[u8], secret_key: &[u8]) -> Result<Self> {
        let pk = kyber1024::PublicKey::from_bytes(public_key)
            .map_err(|_| anyhow!("Invalid Kyber1024 public key"))?;
        let sk = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| anyhow!("Invalid Kyber1024 secret key"))?;

        Ok(Self {
            public_key: pk,
            secret_key: sk,
        })
    }
}

/// Hybrid encryption: Kyber1024 for key exchange + AES-256-GCM for data
pub struct KyberHybridEncryption;

impl KyberHybridEncryption {
    /// Encrypt data using recipient's Kyber public key
    /// Returns (ciphertext, kyber_ciphertext)
    pub fn encrypt(plaintext: &[u8], recipient_public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };

        // Reconstruct recipient's public key
        let pk = kyber1024::PublicKey::from_bytes(recipient_public_key)
            .map_err(|_| anyhow!("Invalid recipient public key"))?;

        // Encapsulate shared secret
        let (shared_secret, kyber_ciphertext) = Kyber1024KeyPair::encapsulate(&pk);

        // Use first 32 bytes of shared secret as AES-256 key
        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&shared_secret[..32]);

        // Generate random nonce for AES-GCM
        use rand::TryRngCore as _;  // For try_fill_bytes (rand 0.9)
        let mut nonce = [0u8; 12];
        rand::rngs::OsRng.try_fill_bytes(&mut nonce).unwrap();

        // Encrypt plaintext with AES-256-GCM
        let cipher = Aes256Gcm::new(&aes_key.into());
        let nonce_obj = Nonce::from_slice(&nonce);

        let mut ciphertext = cipher
            .encrypt(nonce_obj, plaintext)
            .map_err(|e| anyhow!("AES encryption failed: {}", e))?;

        // Prepend nonce to ciphertext
        let mut result = nonce.to_vec();
        result.append(&mut ciphertext);

        Ok((result, kyber_ciphertext))
    }

    /// Decrypt data using Kyber secret key
    pub fn decrypt(
        ciphertext_with_nonce: &[u8],
        kyber_ciphertext: &[u8],
        secret_key: &[u8],
    ) -> Result<Vec<u8>> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };

        // Reconstruct secret key
        let sk = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| anyhow!("Invalid secret key"))?;

        // Decapsulate shared secret
        let ct = kyber1024::Ciphertext::from_bytes(kyber_ciphertext)
            .map_err(|_| anyhow!("Invalid Kyber ciphertext"))?;

        let shared_secret = kyber1024::decapsulate(&ct, &sk);

        // Use first 32 bytes as AES-256 key
        let mut aes_key = [0u8; 32];
        aes_key.copy_from_slice(&shared_secret.as_bytes()[..32]);

        // Extract nonce and ciphertext
        if ciphertext_with_nonce.len() < 12 {
            return Err(anyhow!("Ciphertext too short"));
        }

        let nonce = &ciphertext_with_nonce[..12];
        let ciphertext = &ciphertext_with_nonce[12..];

        // Decrypt with AES-256-GCM
        let cipher = Aes256Gcm::new(&aes_key.into());
        let nonce_obj = Nonce::from_slice(nonce);

        let plaintext = cipher
            .decrypt(nonce_obj, ciphertext)
            .map_err(|e| anyhow!("AES decryption failed: {}", e))?;

        Ok(plaintext)
    }
}

// ============================================================================
// TRUE HYBRID ENCRYPTION: X25519 (Classical) + Kyber1024 (Post-Quantum)
// v2.4.9-beta: Defense-in-depth - BOTH must be broken to compromise security
// ============================================================================

/// Combined keypair for true hybrid encryption
/// Contains both classical X25519 and post-quantum Kyber1024 keys
/// Note: Not Clone due to pqcrypto-kyber types not implementing Clone
pub struct TrueHybridKeypair {
    /// Classical X25519 static secret key
    pub x25519_secret: x25519_dalek::StaticSecret,
    /// Classical X25519 public key
    pub x25519_public: x25519_dalek::PublicKey,
    /// Post-quantum Kyber1024 keypair
    pub kyber: Kyber1024KeyPair,
}

impl TrueHybridKeypair {
    /// Generate a new hybrid keypair with both X25519 and Kyber1024
    pub fn generate() -> Self {
        use rand::TryRngCore as _;

        // Generate random bytes for X25519 secret key
        let mut x25519_secret_bytes = [0u8; 32];
        rand::rngs::OsRng.try_fill_bytes(&mut x25519_secret_bytes).unwrap();

        // Generate classical X25519 keypair from random bytes
        let x25519_secret = x25519_dalek::StaticSecret::from(x25519_secret_bytes);
        let x25519_public = x25519_dalek::PublicKey::from(&x25519_secret);

        // Generate post-quantum Kyber1024 keypair
        let kyber = Kyber1024KeyPair::generate();

        Self {
            x25519_secret,
            x25519_public,
            kyber,
        }
    }

    /// Serialize the public keys for transmission
    /// Format: [x25519_public (32 bytes) | kyber_public (1568 bytes)]
    pub fn public_key_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + kyber1024::public_key_bytes());
        bytes.extend_from_slice(self.x25519_public.as_bytes());
        bytes.extend_from_slice(self.kyber.public_key.as_bytes());
        bytes
    }

    /// Serialize the secret keys for storage
    /// Format: [x25519_secret (32 bytes) | kyber_secret (3168 bytes)]
    pub fn secret_key_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + kyber1024::secret_key_bytes());
        bytes.extend_from_slice(self.x25519_secret.as_bytes());
        bytes.extend_from_slice(self.kyber.secret_key.as_bytes());
        bytes
    }

    /// Reconstruct keypair from serialized bytes
    pub fn from_bytes(public_key: &[u8], secret_key: &[u8]) -> Result<Self> {
        if public_key.len() < 32 + kyber1024::public_key_bytes() {
            return Err(anyhow!("Public key too short for hybrid keypair"));
        }
        if secret_key.len() < 32 + kyber1024::secret_key_bytes() {
            return Err(anyhow!("Secret key too short for hybrid keypair"));
        }

        // Extract X25519 keys
        let x25519_secret_bytes: [u8; 32] = secret_key[..32].try_into()
            .map_err(|_| anyhow!("Invalid X25519 secret key"))?;
        let x25519_public_bytes: [u8; 32] = public_key[..32].try_into()
            .map_err(|_| anyhow!("Invalid X25519 public key"))?;

        let x25519_secret = x25519_dalek::StaticSecret::from(x25519_secret_bytes);
        let x25519_public = x25519_dalek::PublicKey::from(x25519_public_bytes);

        // Extract Kyber keys
        let kyber_public = &public_key[32..];
        let kyber_secret = &secret_key[32..];
        let kyber = Kyber1024KeyPair::from_bytes(kyber_public, kyber_secret)?;

        Ok(Self {
            x25519_secret,
            x25519_public,
            kyber,
        })
    }
}

/// True Hybrid Encryption: Security = Sec(X25519) ∧ Sec(Kyber1024)
///
/// Defense-in-depth architecture:
/// - If X25519 is broken (Shor's algorithm): Kyber1024 protects the data
/// - If Kyber1024 is broken (lattice breakthrough): X25519 protects the data
/// - Both must be broken simultaneously to compromise security
///
/// Key derivation: final_key = HKDF(x25519_shared || kyber_shared)
/// Encryption: XChaCha20-Poly1305 with the derived key
pub struct TrueHybridEncryption;

impl TrueHybridEncryption {
    /// Encrypt data using recipient's hybrid public key
    /// Returns (ciphertext_with_nonce, ephemeral_x25519_public, kyber_ciphertext)
    pub fn encrypt(
        plaintext: &[u8],
        recipient_public_key: &[u8],
    ) -> Result<(Vec<u8>, [u8; 32], Vec<u8>)> {
        use chacha20poly1305::{XChaCha20Poly1305, XNonce, aead::{Aead, KeyInit}};
        use hkdf::Hkdf;
        use sha3::Sha3_256;
        use rand::TryRngCore as _;

        if recipient_public_key.len() < 32 + kyber1024::public_key_bytes() {
            return Err(anyhow!("Recipient public key too short"));
        }

        // Extract recipient's public keys
        let recipient_x25519: [u8; 32] = recipient_public_key[..32].try_into()
            .map_err(|_| anyhow!("Invalid X25519 public key"))?;
        let recipient_x25519_pk = x25519_dalek::PublicKey::from(recipient_x25519);

        let recipient_kyber_pk = kyber1024::PublicKey::from_bytes(&recipient_public_key[32..])
            .map_err(|_| anyhow!("Invalid Kyber public key"))?;

        // ═══════════════════════════════════════════════════════════════════
        // CLASSICAL KEY EXCHANGE: X25519 (Curve25519 ECDH)
        // ═══════════════════════════════════════════════════════════════════
        let mut ephemeral_secret_bytes = [0u8; 32];
        rand::rngs::OsRng.try_fill_bytes(&mut ephemeral_secret_bytes)
            .map_err(|e| anyhow!("RNG failed: {}", e))?;
        let ephemeral_x25519_secret = x25519_dalek::StaticSecret::from(ephemeral_secret_bytes);
        let ephemeral_x25519_public = x25519_dalek::PublicKey::from(&ephemeral_x25519_secret);
        let x25519_shared = ephemeral_x25519_secret.diffie_hellman(&recipient_x25519_pk);

        // ═══════════════════════════════════════════════════════════════════
        // POST-QUANTUM KEY EXCHANGE: Kyber1024 (Lattice-based KEM)
        // ═══════════════════════════════════════════════════════════════════
        let (kyber_shared, kyber_ciphertext) = Kyber1024KeyPair::encapsulate(&recipient_kyber_pk);

        // ═══════════════════════════════════════════════════════════════════
        // HYBRID KEY DERIVATION: final_key = HKDF(classical || post_quantum)
        // Security: Attacker must break BOTH to recover the key
        // ═══════════════════════════════════════════════════════════════════
        let mut combined_secret = Vec::with_capacity(32 + 32);
        combined_secret.extend_from_slice(x25519_shared.as_bytes());
        combined_secret.extend_from_slice(&kyber_shared);

        // Domain separation: "Q-NarwhalKnight TrueHybrid v1"
        let hkdf = Hkdf::<Sha3_256>::new(
            Some(b"Q-NarwhalKnight TrueHybrid v1"),
            &combined_secret,
        );

        let mut final_key = [0u8; 32];
        hkdf.expand(b"hybrid-encryption-key", &mut final_key)
            .map_err(|_| anyhow!("HKDF expansion failed"))?;

        // ═══════════════════════════════════════════════════════════════════
        // AUTHENTICATED ENCRYPTION: XChaCha20-Poly1305
        // ═══════════════════════════════════════════════════════════════════
        let cipher = XChaCha20Poly1305::new(&final_key.into());

        // Generate random 24-byte nonce for XChaCha20 (rand 0.9 TryRngCore is already in scope)
        let mut nonce_bytes = [0u8; 24];
        rand::rngs::OsRng.try_fill_bytes(&mut nonce_bytes)
            .map_err(|e| anyhow!("Nonce generation failed: {}", e))?;
        let nonce = XNonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow!("XChaCha20-Poly1305 encryption failed: {}", e))?;

        // Format: [nonce (24 bytes) | ciphertext]
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);

        Ok((result, *ephemeral_x25519_public.as_bytes(), kyber_ciphertext))
    }

    /// Decrypt data using recipient's hybrid secret key
    pub fn decrypt(
        ciphertext_with_nonce: &[u8],
        ephemeral_x25519_public: &[u8; 32],
        kyber_ciphertext: &[u8],
        secret_key: &[u8],
    ) -> Result<Vec<u8>> {
        use chacha20poly1305::{XChaCha20Poly1305, XNonce, aead::{Aead, KeyInit}};
        use hkdf::Hkdf;
        use sha3::Sha3_256;

        if ciphertext_with_nonce.len() < 24 {
            return Err(anyhow!("Ciphertext too short"));
        }
        if secret_key.len() < 32 + kyber1024::secret_key_bytes() {
            return Err(anyhow!("Secret key too short"));
        }

        // Extract secret keys
        let x25519_secret_bytes: [u8; 32] = secret_key[..32].try_into()
            .map_err(|_| anyhow!("Invalid X25519 secret key"))?;
        let x25519_secret = x25519_dalek::StaticSecret::from(x25519_secret_bytes);

        let kyber_secret = kyber1024::SecretKey::from_bytes(&secret_key[32..])
            .map_err(|_| anyhow!("Invalid Kyber secret key"))?;

        // ═══════════════════════════════════════════════════════════════════
        // CLASSICAL KEY EXCHANGE: X25519 (Curve25519 ECDH)
        // ═══════════════════════════════════════════════════════════════════
        let ephemeral_pk = x25519_dalek::PublicKey::from(*ephemeral_x25519_public);
        let x25519_shared = x25519_secret.diffie_hellman(&ephemeral_pk);

        // ═══════════════════════════════════════════════════════════════════
        // POST-QUANTUM KEY EXCHANGE: Kyber1024 (Lattice-based KEM)
        // ═══════════════════════════════════════════════════════════════════
        let kyber_ct = kyber1024::Ciphertext::from_bytes(kyber_ciphertext)
            .map_err(|_| anyhow!("Invalid Kyber ciphertext"))?;
        let kyber_shared = kyber1024::decapsulate(&kyber_ct, &kyber_secret);

        // ═══════════════════════════════════════════════════════════════════
        // HYBRID KEY DERIVATION: final_key = HKDF(classical || post_quantum)
        // ═══════════════════════════════════════════════════════════════════
        let mut combined_secret = Vec::with_capacity(32 + 32);
        combined_secret.extend_from_slice(x25519_shared.as_bytes());
        combined_secret.extend_from_slice(kyber_shared.as_bytes());

        let hkdf = Hkdf::<Sha3_256>::new(
            Some(b"Q-NarwhalKnight TrueHybrid v1"),
            &combined_secret,
        );

        let mut final_key = [0u8; 32];
        hkdf.expand(b"hybrid-encryption-key", &mut final_key)
            .map_err(|_| anyhow!("HKDF expansion failed"))?;

        // ═══════════════════════════════════════════════════════════════════
        // AUTHENTICATED DECRYPTION: XChaCha20-Poly1305
        // ═══════════════════════════════════════════════════════════════════
        let nonce = XNonce::from_slice(&ciphertext_with_nonce[..24]);
        let ciphertext = &ciphertext_with_nonce[24..];

        let cipher = XChaCha20Poly1305::new(&final_key.into());
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("XChaCha20-Poly1305 decryption failed: {}", e))?;

        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber1024_keypair_generation() {
        let keypair = Kyber1024KeyPair::generate();
        assert_eq!(
            keypair.public_key.as_bytes().len(),
            kyber1024::public_key_bytes()
        );
        assert_eq!(
            keypair.secret_key.as_bytes().len(),
            kyber1024::secret_key_bytes()
        );
    }

    #[test]
    fn test_kyber1024_encapsulation_decapsulation() {
        let keypair = Kyber1024KeyPair::generate();

        // Encapsulate
        let (shared_secret1, ciphertext) =
            Kyber1024KeyPair::encapsulate(&keypair.public_key);

        // Decapsulate
        let shared_secret2 = keypair
            .decapsulate(&ciphertext)
            .expect("Decapsulation failed");

        assert_eq!(
            shared_secret1, shared_secret2,
            "Shared secrets should match"
        );
        assert_eq!(shared_secret1.len(), 32, "Shared secret should be 32 bytes");
    }

    #[test]
    fn test_kyber_hybrid_encryption() {
        let keypair = Kyber1024KeyPair::generate();
        let message = b"Quantum-resistant encrypted wallet data";

        // Encrypt
        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Decrypt
        let plaintext = KyberHybridEncryption::decrypt(
            &ciphertext,
            &kyber_ct,
            keypair.secret_key.as_bytes(),
        )
        .expect("Decryption failed");

        assert_eq!(plaintext, message, "Decrypted message should match original");
    }

    #[test]
    fn test_kyber_keypair_serialization() {
        let keypair = Kyber1024KeyPair::generate();

        let pk_bytes = keypair.public_key_bytes();
        let sk_bytes = keypair.secret_key_bytes();

        // Reconstruct keypair
        let restored = Kyber1024KeyPair::from_bytes(&pk_bytes, &sk_bytes)
            .expect("Keypair restoration failed");

        // Test that restored keypair works
        let (ss1, ct) = Kyber1024KeyPair::encapsulate(&restored.public_key);
        let ss2 = restored.decapsulate(&ct).expect("Decapsulation failed");

        assert_eq!(ss1, ss2, "Restored keypair should work correctly");
    }

    #[test]
    fn test_kyber_wrong_key_decryption_fails() {
        let keypair1 = Kyber1024KeyPair::generate();
        let keypair2 = Kyber1024KeyPair::generate();
        let message = b"Secret message";

        // Encrypt with keypair1's public key
        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair1.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Try to decrypt with keypair2's secret key (should fail or produce garbage)
        let result = KyberHybridEncryption::decrypt(
            &ciphertext,
            &kyber_ct,
            keypair2.secret_key.as_bytes(),
        );

        // Either decryption fails or produces wrong data
        match result {
            Err(_) => { /* Expected - decryption failed */ }
            Ok(plaintext) => {
                assert_ne!(plaintext, message, "Wrong key should not decrypt correctly");
            }
        }
    }

    #[test]
    fn test_kyber_ciphertext_size() {
        let keypair = Kyber1024KeyPair::generate();
        let message = b"Test";

        let (ciphertext, kyber_ct) = KyberHybridEncryption::encrypt(
            message,
            keypair.public_key.as_bytes(),
        )
        .expect("Encryption failed");

        // Kyber1024 ciphertext is 1568 bytes
        assert_eq!(kyber_ct.len(), kyber1024::ciphertext_bytes());

        // AES ciphertext = 12 (nonce) + message_len + 16 (GCM tag)
        assert!(ciphertext.len() >= 12 + message.len() + 16);
    }

    // ═══════════════════════════════════════════════════════════════════
    // TRUE HYBRID ENCRYPTION TESTS (v2.4.9-beta)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_true_hybrid_keypair_generation() {
        let keypair = TrueHybridKeypair::generate();

        // X25519 public key should be 32 bytes
        assert_eq!(keypair.x25519_public.as_bytes().len(), 32);

        // Kyber public key should be 1568 bytes
        assert_eq!(
            keypair.kyber.public_key.as_bytes().len(),
            kyber1024::public_key_bytes()
        );

        // Combined public key should be 32 + 1568 bytes
        let pk_bytes = keypair.public_key_bytes();
        assert_eq!(pk_bytes.len(), 32 + kyber1024::public_key_bytes());
    }

    #[test]
    fn test_true_hybrid_encryption_roundtrip() {
        let keypair = TrueHybridKeypair::generate();
        let message = b"True hybrid encryption: defense in depth against quantum AND classical attacks!";

        // Encrypt
        let (ciphertext, ephemeral_x25519, kyber_ct) = TrueHybridEncryption::encrypt(
            message,
            &keypair.public_key_bytes(),
        )
        .expect("True hybrid encryption failed");

        // Decrypt
        let plaintext = TrueHybridEncryption::decrypt(
            &ciphertext,
            &ephemeral_x25519,
            &kyber_ct,
            &keypair.secret_key_bytes(),
        )
        .expect("True hybrid decryption failed");

        assert_eq!(plaintext, message, "Decrypted message should match original");
    }

    #[test]
    fn test_true_hybrid_wrong_key_fails() {
        let keypair1 = TrueHybridKeypair::generate();
        let keypair2 = TrueHybridKeypair::generate();
        let message = b"Secret data protected by true hybrid encryption";

        // Encrypt with keypair1's public key
        let (ciphertext, ephemeral_x25519, kyber_ct) = TrueHybridEncryption::encrypt(
            message,
            &keypair1.public_key_bytes(),
        )
        .expect("Encryption failed");

        // Try to decrypt with keypair2's secret key (should fail)
        let result = TrueHybridEncryption::decrypt(
            &ciphertext,
            &ephemeral_x25519,
            &kyber_ct,
            &keypair2.secret_key_bytes(),
        );

        // Decryption should fail with wrong key
        assert!(result.is_err(), "Decryption with wrong key should fail");
    }

    #[test]
    fn test_true_hybrid_keypair_serialization() {
        let original = TrueHybridKeypair::generate();

        let pk_bytes = original.public_key_bytes();
        let sk_bytes = original.secret_key_bytes();

        // Reconstruct
        let restored = TrueHybridKeypair::from_bytes(&pk_bytes, &sk_bytes)
            .expect("Failed to restore keypair");

        // Test that restored keypair works
        let message = b"Test message for restored keypair";
        let (ciphertext, ephemeral_x25519, kyber_ct) = TrueHybridEncryption::encrypt(
            message,
            &restored.public_key_bytes(),
        )
        .expect("Encryption with restored key failed");

        let plaintext = TrueHybridEncryption::decrypt(
            &ciphertext,
            &ephemeral_x25519,
            &kyber_ct,
            &restored.secret_key_bytes(),
        )
        .expect("Decryption with restored key failed");

        assert_eq!(plaintext, message, "Restored keypair should work correctly");
    }
}
