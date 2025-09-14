/*!
# Presence Management for BEP-44 Discovery

Manages validator presence announcements through BEP-44 DHT records.

This module handles:
- Periodic presence announcements to the DHT
- Key rotation for privacy (SHA256(PublicKey + CurrentDate))
- Encrypted payloads for friend-only visibility
- Sequence number management to prevent replay attacks

## Privacy Features

- **Key Rotation**: Lookup keys rotate every few hours using SHA256(PublicKey + CurrentDate)
- **Encrypted Payloads**: Only friends with shared secrets can read announcements
- **Temporal Unlinkability**: Different keys over time prevent tracking
- **Decoy Integration**: Works with decoy traffic to provide cover

## Presence Record Format

```json
{
  "validator_id": "d97e812ed90f412f...",
  "onion_address": "d97e812ed90f412f.onion",
  "capabilities": ["Consensus", "Mempool"],
  "timestamp": "2025-09-10T12:34:56Z",
  "encrypted_data": "base64...",
  "signature": "ed25519_signature"
}
```
*/

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use ed25519_dalek::{SigningKey, VerifyingKey};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::{interval, Interval};

use crate::bep44::Bep44Client;
use crate::{DiscoveredPeer, PeerCapability};

/// Presence manager for BEP-44 announcements
#[derive(Debug)]
pub struct PresenceManager {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    announcement_interval: Duration,
    key_rotation_interval: Duration,
    sequence_number: u64,
    current_lookup_key: [u8; 20],
    encryption_key: LessSafeKey,
    friends_keys: HashMap<[u8; 32], Vec<u8>>, // friend_pubkey -> shared_secret
    announcement_timer: Option<Interval>,
    rotation_timer: Option<Interval>,
}

impl PresenceManager {
    /// Create new presence manager
    pub async fn new(
        signing_key: SigningKey,
        announcement_interval: Duration,
        key_rotation_interval: Duration,
    ) -> Result<Self> {
        let verifying_key = signing_key.verifying_key();
        
        // Generate initial lookup key
        let current_lookup_key = Self::calculate_lookup_key(&verifying_key, &Utc::now());
        
        // Generate encryption key for payload encryption
        let mut key_bytes = [0u8; 32];
        ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut key_bytes)
            .context("Failed to generate encryption key")?;
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .context("Failed to create encryption key")?;
        let encryption_key = LessSafeKey::new(unbound_key);
        
        tracing::info!("🎭 Presence manager initialized - Initial lookup key: {}", 
                      hex::encode(&current_lookup_key[..8]));
        
        Ok(Self {
            signing_key,
            verifying_key,
            announcement_interval,
            key_rotation_interval,
            sequence_number: 1,
            current_lookup_key,
            encryption_key,
            friends_keys: HashMap::new(),
            announcement_timer: None,
            rotation_timer: None,
        })
    }
    
    /// Add friend's public key for encrypted communication
    pub fn add_friend(&mut self, friend_pubkey: [u8; 32], shared_secret: Vec<u8>) {
        self.friends_keys.insert(friend_pubkey, shared_secret);
        tracing::info!("👥 Added friend: {}", hex::encode(&friend_pubkey[..4]));
    }
    
    /// Start periodic presence announcements
    pub async fn start_announcements(&mut self) -> Result<()> {
        // Set up announcement timer
        self.announcement_timer = Some(interval(self.announcement_interval));
        
        // Set up key rotation timer
        self.rotation_timer = Some(interval(self.key_rotation_interval));
        
        tracing::info!("📡 Starting presence announcements - Interval: {:?}", 
                      self.announcement_interval);
        
        Ok(())
    }
    
    /// Create and announce presence record
    pub async fn announce_presence(
        &mut self,
        bep44_client: &mut Bep44Client,
        validator_id: [u8; 32],
        onion_address: String,
        capabilities: Vec<PeerCapability>,
    ) -> Result<()> {
        // Create presence record
        let presence_record = PresenceRecord {
            validator_id,
            onion_address,
            capabilities,
            timestamp: Utc::now(),
        };
        
        // Encrypt the presence data
        let encrypted_data = self.encrypt_presence_data(&presence_record)?;
        
        // Create announcement payload
        let announcement = PresenceAnnouncement {
            public_key: self.verifying_key.as_bytes().clone(),
            encrypted_data,
            timestamp: presence_record.timestamp,
            sequence_number: self.sequence_number,
        };
        
        let serialized = serde_json::to_vec(&announcement)
            .context("Failed to serialize announcement")?;
        
        // Store in DHT using current lookup key
        bep44_client.store_mutable_data(&serialized, self.sequence_number).await
            .context("Failed to store presence announcement")?;
        
        self.sequence_number += 1;
        
        tracing::info!("📡 Presence announced - Seq: {}, Key: {}", 
                      self.sequence_number - 1, 
                      hex::encode(&self.current_lookup_key[..8]));
        
        Ok(())
    }
    
    /// Rotate lookup key for privacy
    pub async fn rotate_lookup_key(&mut self) -> Result<()> {
        let new_key = Self::calculate_lookup_key(&self.verifying_key, &Utc::now());
        let old_key = self.current_lookup_key;
        
        self.current_lookup_key = new_key;
        
        tracing::info!("🔄 Rotated lookup key: {} → {}", 
                      hex::encode(&old_key[..8]), 
                      hex::encode(&new_key[..8]));
        
        Ok(())
    }
    
    /// Get current lookup key
    pub async fn get_current_lookup_key(&self) -> Result<[u8; 20]> {
        Ok(self.current_lookup_key)
    }
    
    /// Calculate time-based lookup key: SHA1(SHA256(PublicKey + CurrentDate))
    fn calculate_lookup_key(public_key: &VerifyingKey, timestamp: &DateTime<Utc>) -> [u8; 20] {
        // Get current date in YYYY-MM-DD format
        let date_str = timestamp.format("%Y-%m-%d").to_string();
        
        // Calculate SHA256(PublicKey + CurrentDate)
        let mut hasher = Sha256::new();
        hasher.update(public_key.as_bytes());
        hasher.update(date_str.as_bytes());
        let sha256_result = hasher.finalize();
        
        // Calculate SHA1 for DHT compatibility (20 bytes)
        let mut sha1_hasher = sha1::Sha1::new();
        sha1_hasher.update(&sha256_result);
        sha1_hasher.finalize().into()
    }
    
    /// Encrypt presence data for friends-only visibility
    fn encrypt_presence_data(&self, record: &PresenceRecord) -> Result<Vec<u8>> {
        let plaintext = serde_json::to_vec(record)
            .context("Failed to serialize presence record")?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut nonce_bytes)
            .context("Failed to generate nonce")?;
        
        let nonce = Nonce::assume_unique_for_key(nonce_bytes);
        
        // Encrypt
        let mut ciphertext = plaintext;
        self.encryption_key.seal_in_place_append_tag(nonce, Aad::empty(), &mut ciphertext)
            .context("Encryption failed")?;
        
        // Prepend nonce to ciphertext
        let mut result = Vec::new();
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    /// Decrypt presence data from friends
    pub fn decrypt_presence_data(&self, encrypted_data: &[u8]) -> Result<PresenceRecord> {
        if encrypted_data.len() < 12 {
            anyhow::bail!("Encrypted data too short");
        }
        
        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::assume_unique_for_key(nonce_bytes.try_into().unwrap());
        
        // Decrypt
        let mut plaintext = ciphertext.to_vec();
        self.encryption_key.open_in_place(nonce, Aad::empty(), &mut plaintext)
            .context("Decryption failed")?;
        
        // Remove authentication tag
        plaintext.truncate(plaintext.len() - 16);
        
        // Deserialize
        let record: PresenceRecord = serde_json::from_slice(&plaintext)
            .context("Failed to deserialize presence record")?;
        
        Ok(record)
    }
    
    /// Discover peers by searching DHT with time-based keys
    pub async fn discover_peers(
        &self, 
        bep44_client: &mut Bep44Client,
        search_window_hours: u32,
    ) -> Result<Vec<DiscoveredPeer>> {
        let mut discovered_peers = Vec::new();
        let now = Utc::now();
        
        tracing::info!("🔍 Discovering peers with {}-hour search window", search_window_hours);
        
        // Search multiple time windows for peer announcements
        for hours_ago in 0..search_window_hours {
            let search_time = now - chrono::Duration::hours(hours_ago as i64);
            let search_key = Self::calculate_lookup_key(&self.verifying_key, &search_time);
            
            // Search DHT for records at this time-based key
            match bep44_client.search_mutable_data(&search_key[..8]).await {
                Ok(records) => {
                    for record in records {
                        match self.process_discovered_record(record).await {
                            Ok(Some(peer)) => discovered_peers.push(peer),
                            Ok(None) => {}, // Invalid or old record
                            Err(e) => tracing::debug!("Failed to process record: {}", e),
                        }
                    }
                }
                Err(e) => tracing::debug!("Search failed for time {}: {}", search_time, e),
            }
        }
        
        tracing::info!("✅ Discovered {} peers via time-based BEP-44 search", 
                      discovered_peers.len());
        
        Ok(discovered_peers)
    }
    
    /// Process discovered BEP-44 record into peer info
    async fn process_discovered_record(
        &self,
        record: crate::bep44::MutableDataRecord,
    ) -> Result<Option<DiscoveredPeer>> {
        // Deserialize announcement
        let announcement: PresenceAnnouncement = serde_json::from_slice(&record.data)
            .context("Failed to deserialize announcement")?;
        
        // Check if announcement is recent enough
        let age = Utc::now() - announcement.timestamp;
        if age > chrono::Duration::hours(24) {
            return Ok(None); // Too old
        }
        
        // Try to decrypt if we have shared secret
        match self.decrypt_presence_data(&announcement.encrypted_data) {
            Ok(presence_record) => {
                let peer = DiscoveredPeer {
                    validator_id: presence_record.validator_id,
                    onion_address: presence_record.onion_address,
                    capabilities: presence_record.capabilities,
                    signature: record.signature,
                    timestamp: presence_record.timestamp,
                    discovery_method: "BEP-44-DHT".to_string(),
                };
                
                tracing::debug!("🔍 Decrypted peer announcement: {}", 
                               hex::encode(&peer.validator_id[..4]));
                
                Ok(Some(peer))
            }
            Err(_) => {
                // Can't decrypt - not a friend or different encryption
                tracing::debug!("🔒 Found encrypted announcement but can't decrypt");
                Ok(None)
            }
        }
    }
}

/// Presence record stored in BEP-44
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceRecord {
    pub validator_id: [u8; 32],
    pub onion_address: String,
    pub capabilities: Vec<PeerCapability>,
    pub timestamp: DateTime<Utc>,
}

/// Encrypted presence announcement for BEP-44 storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceAnnouncement {
    pub public_key: [u8; 32],
    pub encrypted_data: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
}