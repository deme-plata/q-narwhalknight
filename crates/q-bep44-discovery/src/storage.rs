use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{PublicKey, Signature};
use sled::Db;
use std::path::Path;
use tracing::{debug, info, warn};

/// DHT storage abstraction for immutable and mutable data
#[async_trait]
pub trait DhtStorage: Send + Sync {
    /// Store immutable data with SHA-1 key
    async fn store_immutable(&self, key: &[u8], value: Vec<u8>) -> Result<()>;

    /// Store mutable data with Ed25519 public key and sequence number
    async fn store_mutable(
        &self,
        pubkey: &PublicKey,
        seq: u64,
        value: Vec<u8>,
        signature: &Signature,
    ) -> Result<()>;

    /// Retrieve data by key (works for both immutable and mutable)
    async fn retrieve(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Retrieve mutable data by public key (returns latest sequence)
    async fn retrieve_mutable(&self, pubkey: &PublicKey) -> Result<Option<(u64, Vec<u8>, Signature)>>;

    /// Get statistics about stored data
    async fn stats(&self) -> Result<StorageStats>;
}

/// Storage statistics for monitoring
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub immutable_count: u64,
    pub mutable_count: u64,
    pub total_size_bytes: u64,
}

/// Sled-based storage implementation for DHT data
pub struct SledStorage {
    db: Db,
    immutable_tree: sled::Tree,
    mutable_tree: sled::Tree,
}

impl SledStorage {
    /// Create new Sled storage with specified path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path.as_ref())?;

        let immutable_tree = db.open_tree("dht_immutable")?;
        let mutable_tree = db.open_tree("dht_mutable")?;

        info!("Opened Sled DHT storage at: {}", path.as_ref().display());

        Ok(Self {
            db,
            immutable_tree,
            mutable_tree,
        })
    }

    /// Create temporary in-memory storage for testing
    pub fn memory() -> Result<Self> {
        let config = sled::Config::new().temporary(true);
        let db = config.open()?;

        let immutable_tree = db.open_tree("dht_immutable")?;
        let mutable_tree = db.open_tree("dht_mutable")?;

        debug!("Created temporary in-memory DHT storage");

        Ok(Self {
            db,
            immutable_tree,
            mutable_tree,
        })
    }

    /// Flush all pending writes to disk
    pub async fn flush(&self) -> Result<()> {
        self.db.flush_async().await?;
        Ok(())
    }
}

#[async_trait]
impl DhtStorage for SledStorage {
    async fn store_immutable(&self, key: &[u8], value: Vec<u8>) -> Result<()> {
        let db_key = format!("imm:{}", hex::encode(key));

        self.immutable_tree.insert(db_key.as_bytes(), value.as_slice())?;

        debug!("Stored immutable data: {} bytes with key {}",
               value.len(), hex::encode(key));

        Ok(())
    }

    async fn store_mutable(
        &self,
        pubkey: &PublicKey,
        seq: u64,
        value: Vec<u8>,
        signature: &Signature,
    ) -> Result<()> {
        let key = format!("mut:{}", hex::encode(pubkey.as_bytes()));

        // Create storage record with sequence, value, and signature
        let mut record = Vec::new();
        record.extend_from_slice(&seq.to_be_bytes()); // 8 bytes
        record.extend_from_slice(&(value.len() as u32).to_be_bytes()); // 4 bytes
        record.extend_from_slice(&value); // variable length
        record.extend_from_slice(signature.as_bytes()); // 64 bytes

        // Only store if sequence number is newer
        if let Ok(Some(existing)) = self.mutable_tree.get(key.as_bytes()) {
            if existing.len() >= 8 {
                let existing_seq = u64::from_be_bytes([
                    existing[0], existing[1], existing[2], existing[3],
                    existing[4], existing[5], existing[6], existing[7],
                ]);

                if seq <= existing_seq {
                    debug!("Rejecting mutable update: seq {} <= existing {}", seq, existing_seq);
                    return Ok(());
                }
            }
        }

        self.mutable_tree.insert(key.as_bytes(), record)?;

        debug!("Stored mutable data: {} bytes, seq {}, key {}",
               value.len(), seq, hex::encode(pubkey.as_bytes()));

        Ok(())
    }

    async fn retrieve(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let imm_key = format!("imm:{}", hex::encode(key));

        if let Some(data) = self.immutable_tree.get(imm_key.as_bytes())? {
            debug!("Retrieved immutable data: {} bytes", data.len());
            return Ok(Some(data.to_vec()));
        }

        // Try as mutable key (public key lookup)
        let mut_key = format!("mut:{}", hex::encode(key));

        if let Some(record) = self.mutable_tree.get(mut_key.as_bytes())? {
            if record.len() >= 12 { // seq(8) + len(4)
                let value_len = u32::from_be_bytes([
                    record[8], record[9], record[10], record[11],
                ]) as usize;

                if record.len() >= 12 + value_len {
                    let value = record[12..12+value_len].to_vec();
                    debug!("Retrieved mutable data: {} bytes", value.len());
                    return Ok(Some(value));
                }
            }
        }

        Ok(None)
    }

    async fn retrieve_mutable(&self, pubkey: &PublicKey) -> Result<Option<(u64, Vec<u8>, Signature)>> {
        let key = format!("mut:{}", hex::encode(pubkey.as_bytes()));

        if let Some(record) = self.mutable_tree.get(key.as_bytes())? {
            if record.len() >= 76 { // seq(8) + len(4) + min_value(0) + sig(64)
                let seq = u64::from_be_bytes([
                    record[0], record[1], record[2], record[3],
                    record[4], record[5], record[6], record[7],
                ]);

                let value_len = u32::from_be_bytes([
                    record[8], record[9], record[10], record[11],
                ]) as usize;

                if record.len() >= 12 + value_len + 64 {
                    let value = record[12..12+value_len].to_vec();
                    let sig_bytes = &record[12+value_len..12+value_len+64];

                    let signature = Signature::from_bytes(sig_bytes.try_into().unwrap_or([0u8; 64]))?;

                    debug!("Retrieved mutable data: seq {}, {} bytes", seq, value.len());
                    return Ok(Some((seq, value, signature)));
                }
            }
        }

        Ok(None)
    }

    async fn stats(&self) -> Result<StorageStats> {
        let immutable_count = self.immutable_tree.len();
        let mutable_count = self.mutable_tree.len();

        // Estimate total size (Sled doesn't provide exact size easily)
        let total_size_bytes = self.db.size_on_disk()?;

        Ok(StorageStats {
            immutable_count,
            mutable_count,
            total_size_bytes,
        })
    }
}

/// In-memory storage for testing and development
pub struct MemoryStorage {
    immutable: dashmap::DashMap<Vec<u8>, Vec<u8>>,
    mutable: dashmap::DashMap<[u8; 32], (u64, Vec<u8>, Signature)>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            immutable: dashmap::DashMap::new(),
            mutable: dashmap::DashMap::new(),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DhtStorage for MemoryStorage {
    async fn store_immutable(&self, key: &[u8], value: Vec<u8>) -> Result<()> {
        self.immutable.insert(key.to_vec(), value);
        debug!("Stored immutable data in memory: key {}", hex::encode(key));
        Ok(())
    }

    async fn store_mutable(
        &self,
        pubkey: &PublicKey,
        seq: u64,
        value: Vec<u8>,
        signature: &Signature,
    ) -> Result<()> {
        let key = pubkey.as_bytes();

        // Only store if sequence is newer
        if let Some(existing) = self.mutable.get(key) {
            if seq <= existing.0 {
                debug!("Rejecting mutable update: seq {} <= existing {}", seq, existing.0);
                return Ok(());
            }
        }

        self.mutable.insert(*key, (seq, value, *signature));
        debug!("Stored mutable data in memory: seq {}", seq);
        Ok(())
    }

    async fn retrieve(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Try immutable first
        if let Some(value) = self.immutable.get(key) {
            return Ok(Some(value.clone()));
        }

        // Try mutable (key is public key)
        if key.len() == 32 {
            let pubkey_array: [u8; 32] = key.try_into().unwrap_or([0u8; 32]);
            if let Some(entry) = self.mutable.get(&pubkey_array) {
                return Ok(Some(entry.1.clone()));
            }
        }

        Ok(None)
    }

    async fn retrieve_mutable(&self, pubkey: &PublicKey) -> Result<Option<(u64, Vec<u8>, Signature)>> {
        let key = pubkey.as_bytes();

        if let Some(entry) = self.mutable.get(key) {
            let (seq, value, signature) = entry.value();
            Ok(Some((*seq, value.clone(), *signature)))
        } else {
            Ok(None)
        }
    }

    async fn stats(&self) -> Result<StorageStats> {
        let immutable_count = self.immutable.len() as u64;
        let mutable_count = self.mutable.len() as u64;

        // Rough size estimate
        let immutable_size: usize = self.immutable
            .iter()
            .map(|entry| entry.key().len() + entry.value().len())
            .sum();
        let mutable_size: usize = self.mutable
            .iter()
            .map(|entry| 32 + entry.value().1.len() + 64) // pubkey + value + signature
            .sum();

        Ok(StorageStats {
            immutable_count,
            mutable_count,
            total_size_bytes: (immutable_size + mutable_size) as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_sled_immutable_storage() -> Result<()> {
        let storage = SledStorage::memory()?;

        let key = [0x42u8; 20];
        let value = b"test data".to_vec();

        storage.store_immutable(&key, value.clone()).await?;
        let retrieved = storage.retrieve(&key).await?;

        assert_eq!(retrieved, Some(value));
        Ok(())
    }

    #[tokio::test]
    async fn test_sled_mutable_storage() -> Result<()> {
        let storage = SledStorage::memory()?;
        let keypair = Keypair::generate(&mut OsRng);

        let value1 = b"first value".to_vec();
        let value2 = b"second value".to_vec();

        // Create signatures (simplified - in reality would use proper BEP-44 message format)
        let sig1 = keypair.sign(&value1);
        let sig2 = keypair.sign(&value2);

        // Store first version
        storage.store_mutable(&keypair.public, 1, value1.clone(), &sig1).await?;

        let retrieved = storage.retrieve_mutable(&keypair.public).await?;
        assert_eq!(retrieved.as_ref().map(|r| &r.1), Some(&value1));
        assert_eq!(retrieved.as_ref().map(|r| r.0), Some(1));

        // Store second version (higher sequence)
        storage.store_mutable(&keypair.public, 2, value2.clone(), &sig2).await?;

        let retrieved = storage.retrieve_mutable(&keypair.public).await?;
        assert_eq!(retrieved.as_ref().map(|r| &r.1), Some(&value2));
        assert_eq!(retrieved.as_ref().map(|r| r.0), Some(2));

        // Try to store older version (should be rejected)
        storage.store_mutable(&keypair.public, 1, b"old value".to_vec(), &sig1).await?;

        let retrieved = storage.retrieve_mutable(&keypair.public).await?;
        assert_eq!(retrieved.as_ref().map(|r| &r.1), Some(&value2)); // Still second value

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_storage() -> Result<()> {
        let storage = MemoryStorage::new();

        let key = [0x42u8; 20];
        let value = b"test data".to_vec();

        storage.store_immutable(&key, value.clone()).await?;
        let retrieved = storage.retrieve(&key).await?;

        assert_eq!(retrieved, Some(value));

        let stats = storage.stats().await?;
        assert_eq!(stats.immutable_count, 1);
        assert_eq!(stats.mutable_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_stats() -> Result<()> {
        let storage = SledStorage::memory()?;

        // Store some test data
        storage.store_immutable(&[0x01; 20], b"data1".to_vec()).await?;
        storage.store_immutable(&[0x02; 20], b"data2".to_vec()).await?;

        let keypair = Keypair::generate(&mut OsRng);
        let sig = keypair.sign(b"mutable data");
        storage.store_mutable(&keypair.public, 1, b"mutable data".to_vec(), &sig).await?;

        let stats = storage.stats().await?;
        assert_eq!(stats.immutable_count, 2);
        assert_eq!(stats.mutable_count, 1);
        assert!(stats.total_size_bytes > 0);

        Ok(())
    }
}