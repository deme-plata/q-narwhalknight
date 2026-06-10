/*!
# Cryptographic utilities for BEP-44 Discovery (Simplified)

Provides simplified encryption, key rotation, and signature verification for secure peer discovery.

This is a simplified version that demonstrates the architecture without complex ring crypto.
A production version would use proper AEAD encryption.

Key features:
- **Ed25519 Signatures**: For BEP-44 record authenticity
- **XOR Encryption**: Simple encryption for demonstration (not production-ready)
- **Key Derivation**: Time-based key rotation using SHA256
- **Shared Secrets**: Simple shared secret management

## Security Model (Simplified)

1. **Public Signatures**: All BEP-44 records are signed with Ed25519
2. **XOR Encryption**: Simple encryption for proof of concept
3. **Forward Secrecy**: Key rotation every hour prevents long-term correlation  
4. **Deniability**: Decoy traffic provides plausible deniability

## Key Rotation Schedule

```
Hour 0: Key = SHA256(PublicKey + "2025-09-10-00")
Hour 1: Key = SHA256(PublicKey + "2025-09-10-01")  
Hour 2: Key = SHA256(PublicKey + "2025-09-10-02")
...
```

This allows friends to calculate the current lookup key without coordination.
*/

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, SigningKey, VerifyingKey, Signer, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Simplified cryptographic manager for BEP-44 discovery
#[derive(Debug)]
pub struct CryptoManager {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    master_secret: [u8; 32],
    friend_secrets: HashMap<[u8; 32], [u8; 32]>, // friend_pubkey -> shared_secret
}

impl CryptoManager {
    /// Create new crypto manager
    pub fn new(signing_key: SigningKey) -> Result<Self> {
        let verifying_key = signing_key.verifying_key();
        
        // Generate master secret for encryption
        let mut master_secret = [0u8; 32];
        getrandom::getrandom(&mut master_secret)
            .context("Failed to generate master secret")?;
        
        tracing::info!("🔐 Crypto manager initialized - Public key: {}", 
                      hex::encode(verifying_key.as_bytes()));
        
        Ok(Self {
            signing_key,
            verifying_key,
            master_secret,
            friend_secrets: HashMap::new(),
        })
    }
    
    /// Add friend for encrypted communication
    pub fn add_friend(&mut self, friend_public_key: [u8; 32], shared_secret: [u8; 32]) {
        self.friend_secrets.insert(friend_public_key, shared_secret);
        tracing::info!("👥 Added friend: {}", hex::encode(&friend_public_key[..4]));
    }
    
    /// Sign data with Ed25519 for BEP-44 record
    pub fn sign_data(&self, data: &[u8]) -> Signature {
        self.signing_key.sign(data)
    }
    
    /// Verify Ed25519 signature
    pub fn verify_signature(&self, data: &[u8], signature: &Signature, public_key: &VerifyingKey) -> Result<()> {
        public_key.verify(data, signature)
            .context("Signature verification failed")?;
        Ok(())
    }
    
    /// Encrypt data for friend-only visibility (simplified XOR encryption)
    pub fn encrypt_for_friends(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        // Derive encryption key from master secret
        let encryption_key = self.derive_encryption_key(b"friend-encryption")?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        getrandom::getrandom(&mut nonce_bytes)
            .context("Failed to generate nonce")?;
        
        // Simple XOR encryption (NOT production-ready, just for proof of concept)
        let mut ciphertext = plaintext.to_vec();
        for (i, byte) in ciphertext.iter_mut().enumerate() {
            *byte ^= encryption_key[i % 32] ^ nonce_bytes[i % 12];
        }
        
        // Prepend nonce to ciphertext
        let mut result = Vec::new();
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    /// Decrypt friend data (simplified XOR decryption)
    pub fn decrypt_friend_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            anyhow::bail!("Encrypted data too short");
        }
        
        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce: [u8; 12] = nonce_bytes.try_into().unwrap();
        
        // Derive decryption key
        let decryption_key = self.derive_encryption_key(b"friend-encryption")?;
        
        // Simple XOR decryption
        let mut plaintext = ciphertext.to_vec();
        for (i, byte) in plaintext.iter_mut().enumerate() {
            *byte ^= decryption_key[i % 32] ^ nonce[i % 12];
        }
        
        Ok(plaintext)
    }
    
    /// Calculate time-based lookup key for DHT storage
    pub fn calculate_lookup_key(&self, timestamp: &DateTime<Utc>) -> [u8; 20] {
        // Get current hour in format: YYYY-MM-DD-HH
        let hour_str = timestamp.format("%Y-%m-%d-%H").to_string();
        
        // Calculate SHA256(PublicKey + CurrentHour)
        let mut hasher = Sha256::new();
        hasher.update(self.verifying_key.as_bytes());
        hasher.update(hour_str.as_bytes());
        let sha256_result = hasher.finalize();
        
        // Calculate SHA1 for DHT compatibility (20 bytes)
        let mut sha1_hasher = sha1::Sha1::new();
        sha1_hasher.update(&sha256_result);
        sha1_hasher.finalize().into()
    }
    
    /// Calculate lookup key for a specific friend at a given time
    pub fn calculate_friend_lookup_key(&self, friend_public_key: &[u8; 32], timestamp: &DateTime<Utc>) -> Result<[u8; 20]> {
        let hour_str = timestamp.format("%Y-%m-%d-%H").to_string();
        
        // Calculate SHA256(FriendPublicKey + CurrentHour)
        let mut hasher = Sha256::new();
        hasher.update(friend_public_key);
        hasher.update(hour_str.as_bytes());
        let sha256_result = hasher.finalize();
        
        // Calculate SHA1 for DHT compatibility
        let mut sha1_hasher = sha1::Sha1::new();
        sha1_hasher.update(&sha256_result);
        Ok(sha1_hasher.finalize().into())
    }
    
    /// Generate shared secret with friend using simple key mixing
    pub fn generate_shared_secret(&self, friend_public_key: &[u8; 32]) -> Result<[u8; 32]> {
        // Simple key mixing (not cryptographically secure, just for proof of concept)
        let mut hasher = Sha256::new();
        hasher.update(self.signing_key.as_bytes());
        hasher.update(friend_public_key);
        Ok(hasher.finalize().into())
    }
    
    /// Derive encryption key using SHA256
    fn derive_encryption_key(&self, info: &[u8]) -> Result<[u8; 32]> {
        let mut hasher = Sha256::new();
        hasher.update(&self.master_secret);
        hasher.update(info);
        Ok(hasher.finalize().into())
    }
    
    /// Create BEP-44 compliant signature for mutable data
    pub fn create_bep44_signature(&self, data: &[u8], sequence_number: u64) -> Signature {
        // BEP-44 signature format: sign(data + sequence_number)
        let mut message = Vec::new();
        message.extend_from_slice(data);
        message.extend_from_slice(&sequence_number.to_be_bytes());
        
        self.signing_key.sign(&message)
    }
    
    /// Verify BEP-44 signature
    pub fn verify_bep44_signature(
        &self, 
        data: &[u8], 
        sequence_number: u64,
        signature: &Signature, 
        public_key: &VerifyingKey
    ) -> Result<()> {
        // Reconstruct signed message
        let mut message = Vec::new();
        message.extend_from_slice(data);
        message.extend_from_slice(&sequence_number.to_be_bytes());
        
        public_key.verify(&message, signature)
            .context("BEP-44 signature verification failed")?;
        
        Ok(())
    }
    
    /// Generate key rotation schedule for next 24 hours
    pub fn generate_key_schedule(&self, start_time: DateTime<Utc>) -> Vec<KeyScheduleEntry> {
        let mut schedule = Vec::new();
        
        for hour in 0..24 {
            let time = start_time + chrono::Duration::hours(hour);
            let lookup_key = self.calculate_lookup_key(&time);
            
            schedule.push(KeyScheduleEntry {
                time,
                lookup_key,
                hour_offset: hour,
            });
        }
        
        schedule
    }
    
    /// Get current lookup key
    pub fn get_current_lookup_key(&self) -> [u8; 20] {
        self.calculate_lookup_key(&Utc::now())
    }
    
    /// Get public key bytes
    pub fn get_public_key(&self) -> [u8; 32] {
        *self.verifying_key.as_bytes()
    }
    
    /// Get friend secrets (for testing)
    pub fn get_friend_secrets(&self) -> &HashMap<[u8; 32], [u8; 32]> {
        &self.friend_secrets
    }
}

/// Key rotation schedule entry
#[derive(Debug, Clone)]
pub struct KeyScheduleEntry {
    pub time: DateTime<Utc>,
    pub lookup_key: [u8; 20],
    pub hour_offset: i64,
}

/// Encrypted announcement payload (simplified)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EncryptedPayload {
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub encryption_version: u8,
}

impl EncryptedPayload {
    /// Create new encrypted payload
    pub fn new(plaintext: &[u8], crypto_manager: &CryptoManager) -> Result<Self> {
        let encrypted_data = crypto_manager.encrypt_for_friends(plaintext)?;
        
        if encrypted_data.len() < 12 {
            anyhow::bail!("Invalid encrypted data length");
        }
        
        let nonce: [u8; 12] = encrypted_data[..12].try_into().unwrap();
        let ciphertext = encrypted_data[12..].to_vec();
        
        Ok(Self {
            nonce,
            ciphertext,
            timestamp: Utc::now(),
            encryption_version: 1,
        })
    }
    
    /// Decrypt payload
    pub fn decrypt(&self, crypto_manager: &CryptoManager) -> Result<Vec<u8>> {
        let mut encrypted_data = Vec::new();
        encrypted_data.extend_from_slice(&self.nonce);
        encrypted_data.extend_from_slice(&self.ciphertext);
        
        crypto_manager.decrypt_friend_data(&encrypted_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_key_rotation() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let crypto_manager = CryptoManager::new(signing_key).unwrap();
        
        let time1 = Utc::now();
        let time2 = time1 + chrono::Duration::hours(1);
        
        let key1 = crypto_manager.calculate_lookup_key(&time1);
        let key2 = crypto_manager.calculate_lookup_key(&time2);
        
        assert_ne!(key1, key2, "Keys should be different for different hours");
    }
    
    #[test]
    fn test_encryption_decryption() {
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let crypto_manager = CryptoManager::new(signing_key).unwrap();
        
        let plaintext = b"test message";
        let encrypted = crypto_manager.encrypt_for_friends(plaintext).unwrap();
        let decrypted = crypto_manager.decrypt_friend_data(&encrypted).unwrap();
        
        assert_eq!(plaintext, &decrypted[..]);
    }
}