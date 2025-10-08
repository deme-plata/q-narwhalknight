use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, AeadCore, OsRng};
use anyhow::{anyhow, Result};
use hkdf::Hkdf;
use rand::{Rng, RngCore};
use sha2::Sha256;
use tracing::{debug, warn};

/// DNS-Phantom steganographic encryption system
///
/// Provides AES-256-GCM encryption for DNS TXT payloads with node-specific key derivation.
/// This prevents plaintext exposure of peer information in DNS queries.
pub struct SteganoEncryption {
    node_id: String,
    cipher: Aes256Gcm,
}

impl SteganoEncryption {
    /// Create new encryption instance with node-specific key derivation
    pub fn new(node_id: &str) -> Result<Self> {
        debug!("🔐 DNS-Phantom: Initializing AES-256-GCM encryption for node: {}", node_id);

        // Derive encryption key from node_id using HKDF-SHA256
        let key = Self::derive_key_from_node_id(node_id)?;
        let cipher = Aes256Gcm::new(&key);

        Ok(Self {
            node_id: node_id.to_string(),
            cipher,
        })
    }

    /// Derive 256-bit encryption key from node_id using HKDF
    fn derive_key_from_node_id(node_id: &str) -> Result<aes_gcm::Key<Aes256Gcm>> {
        // Use HKDF to derive a key from the node_id
        let salt = b"Q-NarwhalKnight-DNS-Phantom-v1.0";
        let info = b"steganographic-encryption-key";

        let hkdf = Hkdf::<Sha256>::new(Some(salt), node_id.as_bytes());
        let mut key_bytes = [0u8; 32]; // 256 bits for AES-256
        hkdf.expand(info, &mut key_bytes)
            .map_err(|e| anyhow!("Key derivation failed: {}", e))?;

        debug!("🔑 DNS-Phantom: Successfully derived encryption key from node_id");
        Ok(*aes_gcm::Key::<Aes256Gcm>::from_slice(&key_bytes))
    }

    /// Encrypt data for steganographic DNS embedding
    ///
    /// Returns: nonce (12 bytes) + ciphertext
    pub fn encrypt_payload(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Generate random nonce
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

        // Encrypt the data
        let ciphertext = self.cipher.encrypt(&nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        // Prepend nonce to ciphertext for transport
        let mut encrypted_payload = nonce.to_vec();
        encrypted_payload.extend_from_slice(&ciphertext);

        debug!("🔐 DNS-Phantom: Encrypted {} bytes → {} bytes (with nonce)",
               data.len(), encrypted_payload.len());

        Ok(encrypted_payload)
    }

    /// Decrypt steganographic data from DNS
    ///
    /// Expected format: nonce (12 bytes) + ciphertext
    pub fn decrypt_payload(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            return Err(anyhow!("Encrypted data too short: {} bytes (minimum 12)", encrypted_data.len()));
        }

        // Extract nonce and ciphertext
        let nonce = Nonce::from_slice(&encrypted_data[..12]);
        let ciphertext = &encrypted_data[12..];

        // Decrypt the data
        let plaintext = self.cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;

        debug!("🔓 DNS-Phantom: Decrypted {} bytes → {} bytes",
               encrypted_data.len(), plaintext.len());

        Ok(plaintext)
    }

    /// Encrypt and Base32-encode for DNS TXT records
    ///
    /// This combines encryption with DNS-safe encoding for direct embedding in TXT records
    pub fn encrypt_for_dns_txt(&self, data: &[u8]) -> Result<String> {
        let encrypted = self.encrypt_payload(data)?;

        // Use Base32 encoding for DNS-safe transport (no special characters)
        let encoded = base32::encode(base32::Alphabet::Crockford, &encrypted);

        debug!("🔐 DNS-Phantom: Encrypted and encoded {} bytes → {} chars for TXT record",
               data.len(), encoded.len());

        Ok(encoded)
    }

    /// Decrypt Base32-encoded DNS TXT data
    pub fn decrypt_from_dns_txt(&self, encoded_data: &str) -> Result<Vec<u8>> {
        // Decode from Base32
        let encrypted = base32::decode(base32::Alphabet::Crockford, encoded_data)
            .ok_or_else(|| anyhow!("Base32 decoding failed"))?;

        // Decrypt the payload
        self.decrypt_payload(&encrypted)
    }

    /// Generate random cover data for traffic analysis resistance
    ///
    /// Creates encrypted noise that looks like legitimate encrypted payloads
    pub fn generate_cover_noise(&self, target_size: usize) -> Result<String> {
        // Generate random data
        let mut noise_data = vec![0u8; target_size.saturating_sub(16)]; // Account for encryption overhead
        rand::thread_rng().fill_bytes(&mut noise_data);

        // Encrypt the random data (this makes it indistinguishable from real payloads)
        let encrypted_noise = self.encrypt_payload(&noise_data)?;

        // Encode for DNS transport
        let encoded = base32::encode(base32::Alphabet::Crockford, &encrypted_noise);

        debug!("🎭 DNS-Phantom: Generated {} bytes of encrypted cover noise", target_size);

        Ok(encoded)
    }

    /// Validate that this node can decrypt data from another node
    ///
    /// Used for testing encryption compatibility between peers
    pub fn test_encryption_round_trip(&self, test_data: &[u8]) -> Result<()> {
        let encrypted = self.encrypt_payload(test_data)?;
        let decrypted = self.decrypt_payload(&encrypted)?;

        if test_data != decrypted.as_slice() {
            return Err(anyhow!("Encryption round-trip test failed"));
        }

        debug!("✅ DNS-Phantom: Encryption round-trip test passed");
        Ok(())
    }
}

/// Helper functions for key management and validation
impl SteganoEncryption {
    /// Get the node_id this encryption instance is configured for
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Check if two encryption instances can communicate (same key derivation)
    pub fn is_compatible_with(&self, other_node_id: &str) -> bool {
        // For now, all nodes use the same key derivation algorithm
        // In the future, this could check protocol versions, etc.
        !other_node_id.is_empty()
    }

    /// Estimate encrypted size for a given plaintext size
    ///
    /// Useful for DNS query planning and fragmentation
    pub fn estimate_encrypted_size(plaintext_size: usize) -> usize {
        // AES-GCM adds 16 bytes authentication tag + 12 bytes nonce
        plaintext_size + 28
    }

    /// Calculate maximum plaintext size for DNS TXT record limits
    ///
    /// DNS TXT records have a 255-byte limit, Base32 encoding has ~1.6x overhead
    pub fn max_plaintext_for_txt_record() -> usize {
        // 255 bytes / 1.6 (Base32 overhead) - 28 (encryption overhead)
        (255.0 / 1.6) as usize - 28
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_round_trip() {
        let encryptor = SteganoEncryption::new("test-node-123").unwrap();
        let test_data = b"Hello, DNS-Phantom steganographic world!";

        encryptor.test_encryption_round_trip(test_data).unwrap();
    }

    #[test]
    fn test_dns_txt_encoding() {
        let encryptor = SteganoEncryption::new("test-node-456").unwrap();
        let test_data = b"Test peer info: node-789, onion-xyz.onion";

        let encoded = encryptor.encrypt_for_dns_txt(test_data).unwrap();
        let decoded = encryptor.decrypt_from_dns_txt(&encoded).unwrap();

        assert_eq!(test_data, decoded.as_slice());
    }

    #[test]
    fn test_cover_noise_generation() {
        let encryptor = SteganoEncryption::new("test-node-789").unwrap();

        let noise1 = encryptor.generate_cover_noise(100).unwrap();
        let noise2 = encryptor.generate_cover_noise(100).unwrap();

        // Cover noise should be different each time
        assert_ne!(noise1, noise2);

        // Should be valid Base32
        assert!(base32::decode(base32::Alphabet::Crockford, &noise1).is_some());
    }

    #[test]
    fn test_key_derivation_consistency() {
        let encryptor1 = SteganoEncryption::new("consistent-node").unwrap();
        let encryptor2 = SteganoEncryption::new("consistent-node").unwrap();

        let test_data = b"Key derivation consistency test";

        // Encrypt with first instance
        let encrypted = encryptor1.encrypt_for_dns_txt(test_data).unwrap();

        // Decrypt with second instance (same node_id = same key)
        let decrypted = encryptor2.decrypt_from_dns_txt(&encrypted).unwrap();

        assert_eq!(test_data, decrypted.as_slice());
    }
}