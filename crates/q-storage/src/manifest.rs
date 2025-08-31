use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::Instant,
};
use tracing::{debug, info};

use crate::{kv::KVStore, CF_MANIFEST};

/// Storage manifest for crash recovery and consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageManifest {
    /// Highest contiguous DAG round stored
    pub dag_round_watermark: u64,
    
    /// Height of last finalized block
    pub finalized_height: u64,
    
    /// Last successful checkpoint height
    pub last_checkpoint_height: u64,
    
    /// Total vertices stored
    pub total_vertices: u64,
    
    /// Total payloads stored
    pub total_payloads: u64,
    
    /// Storage format version
    pub version: u32,
    
    /// Last update timestamp
    #[serde(skip)]
    pub last_update: Instant,
}

impl Default for StorageManifest {
    fn default() -> Self {
        Self {
            dag_round_watermark: 0,
            finalized_height: 0,
            last_checkpoint_height: 0,
            total_vertices: 0,
            total_payloads: 0,
            version: 1,
            last_update: Instant::now(),
        }
    }
}

impl StorageManifest {
    const MANIFEST_KEY: &'static [u8] = b"storage_manifest";

    /// Load manifest from storage or create default
    pub async fn load_or_create(kv: &Arc<dyn KVStore>) -> Result<Self> {
        match kv.get(CF_MANIFEST, Self::MANIFEST_KEY).await? {
            Some(data) => {
                let mut manifest: StorageManifest = bincode::deserialize(&data)
                    .context("Failed to deserialize storage manifest")?;
                
                manifest.last_update = Instant::now();
                
                info!("📋 Loaded storage manifest - DAG round: {}, finalized: {}", 
                      manifest.dag_round_watermark, manifest.finalized_height);
                
                Ok(manifest)
            }
            None => {
                info!("📋 Creating new storage manifest");
                let manifest = Self::default();
                
                // Save the new manifest
                manifest.save(kv).await?;
                
                Ok(manifest)
            }
        }
    }

    /// Save manifest to storage
    pub async fn save(&self, kv: &Arc<dyn KVStore>) -> Result<()> {
        let data = bincode::serialize(self)
            .context("Failed to serialize storage manifest")?;
        
        kv.put(CF_MANIFEST, Self::MANIFEST_KEY, &data).await
            .context("Failed to save storage manifest")?;
        
        debug!("💾 Saved storage manifest");
        Ok(())
    }

    /// Update DAG round watermark (highest contiguous round)
    pub async fn update_dag_watermark(&mut self, kv: &Arc<dyn KVStore>, new_round: u64) -> Result<()> {
        if new_round > self.dag_round_watermark {
            debug!("📈 Updating DAG watermark from {} to {}", 
                   self.dag_round_watermark, new_round);
            
            self.dag_round_watermark = new_round;
            self.last_update = Instant::now();
            self.save(kv).await?;
        }
        
        Ok(())
    }

    /// Update finalized height
    pub async fn update_finalized_height(&mut self, kv: &Arc<dyn KVStore>, height: u64) -> Result<()> {
        if height > self.finalized_height {
            debug!("📈 Updating finalized height from {} to {}", 
                   self.finalized_height, height);
            
            self.finalized_height = height;
            self.last_update = Instant::now();
            self.save(kv).await?;
        }
        
        Ok(())
    }

    /// Increment vertex count
    pub async fn increment_vertices(&mut self, kv: &Arc<dyn KVStore>, count: u64) -> Result<()> {
        self.total_vertices += count;
        self.last_update = Instant::now();
        
        // Save every 100 vertices to avoid too frequent writes
        if self.total_vertices % 100 == 0 {
            self.save(kv).await?;
        }
        
        Ok(())
    }

    /// Increment payload count
    pub async fn increment_payloads(&mut self, kv: &Arc<dyn KVStore>, count: u64) -> Result<()> {
        self.total_payloads += count;
        self.last_update = Instant::now();
        
        // Save every 50 payloads to avoid too frequent writes
        if self.total_payloads % 50 == 0 {
            self.save(kv).await?;
        }
        
        Ok(())
    }

    /// Mark checkpoint completion
    pub async fn mark_checkpoint(&mut self, kv: &Arc<dyn KVStore>, height: u64) -> Result<()> {
        info!("📸 Marking checkpoint at height {}", height);
        
        self.last_checkpoint_height = height;
        self.last_update = Instant::now();
        self.save(kv).await?;
        
        Ok(())
    }

    /// Check if storage needs recovery
    pub fn needs_recovery(&self) -> bool {
        // If we have a significant gap between watermark and finalized height,
        // we might need to sync missing data
        let gap = self.finalized_height.saturating_sub(self.dag_round_watermark);
        gap > 100 // Arbitrary threshold
    }

    /// Get recovery information
    pub fn get_recovery_info(&self) -> RecoveryInfo {
        RecoveryInfo {
            dag_round_watermark: self.dag_round_watermark,
            finalized_height: self.finalized_height,
            gap: self.finalized_height.saturating_sub(self.dag_round_watermark),
            last_checkpoint: self.last_checkpoint_height,
            needs_sync: self.needs_recovery(),
        }
    }

    /// Validate manifest consistency
    pub fn validate(&self) -> Result<()> {
        if self.dag_round_watermark > self.finalized_height + 1000 {
            anyhow::bail!("DAG watermark {} far ahead of finalized height {}", 
                         self.dag_round_watermark, self.finalized_height);
        }

        if self.last_checkpoint_height > self.finalized_height {
            anyhow::bail!("Checkpoint height {} ahead of finalized height {}", 
                         self.last_checkpoint_height, self.finalized_height);
        }

        if self.version == 0 {
            anyhow::bail!("Invalid storage version");
        }

        Ok(())
    }

    /// Upgrade manifest version if needed
    pub async fn upgrade_if_needed(&mut self, kv: &Arc<dyn KVStore>) -> Result<bool> {
        const CURRENT_VERSION: u32 = 1;
        
        if self.version < CURRENT_VERSION {
            info!("⬆️ Upgrading storage manifest from v{} to v{}", 
                  self.version, CURRENT_VERSION);
            
            // Perform any necessary migrations
            match self.version {
                0 => {
                    // Migrate from v0 to v1
                    self.total_vertices = 0; // Reset counts
                    self.total_payloads = 0;
                }
                _ => {}
            }
            
            self.version = CURRENT_VERSION;
            self.last_update = Instant::now();
            self.save(kv).await?;
            
            return Ok(true);
        }
        
        Ok(false)
    }
}

/// Recovery information for diagnostics
#[derive(Debug, Clone, Serialize)]
pub struct RecoveryInfo {
    pub dag_round_watermark: u64,
    pub finalized_height: u64,
    pub gap: u64,
    pub last_checkpoint: u64,
    pub needs_sync: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::RocksDBKV;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_manifest_creation() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());
        
        let manifest = StorageManifest::load_or_create(&kv).await.unwrap();
        
        assert_eq!(manifest.dag_round_watermark, 0);
        assert_eq!(manifest.finalized_height, 0);
        assert_eq!(manifest.version, 1);
    }

    #[tokio::test]
    async fn test_manifest_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());
        
        // Create and save manifest
        let mut manifest = StorageManifest::default();
        manifest.dag_round_watermark = 100;
        manifest.finalized_height = 95;
        manifest.save(&kv).await.unwrap();
        
        // Load manifest and verify
        let loaded_manifest = StorageManifest::load_or_create(&kv).await.unwrap();
        assert_eq!(loaded_manifest.dag_round_watermark, 100);
        assert_eq!(loaded_manifest.finalized_height, 95);
    }

    #[tokio::test]
    async fn test_manifest_updates() {
        let temp_dir = TempDir::new().unwrap();
        let kv = Arc::new(RocksDBKV::open_hot_db(temp_dir.path()).await.unwrap());
        
        let mut manifest = StorageManifest::load_or_create(&kv).await.unwrap();
        
        // Test watermark update
        manifest.update_dag_watermark(&kv, 50).await.unwrap();
        assert_eq!(manifest.dag_round_watermark, 50);
        
        // Test finalized height update
        manifest.update_finalized_height(&kv, 45).await.unwrap();
        assert_eq!(manifest.finalized_height, 45);
    }

    #[test]
    fn test_manifest_validation() {
        let mut manifest = StorageManifest::default();
        
        // Valid manifest
        manifest.dag_round_watermark = 100;
        manifest.finalized_height = 98;
        assert!(manifest.validate().is_ok());
        
        // Invalid manifest (watermark too far ahead)
        manifest.dag_round_watermark = 2000;
        manifest.finalized_height = 50;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_recovery_info() {
        let mut manifest = StorageManifest::default();
        manifest.dag_round_watermark = 50;
        manifest.finalized_height = 200;
        
        let recovery_info = manifest.get_recovery_info();
        assert_eq!(recovery_info.gap, 150);
        assert!(recovery_info.needs_sync);
    }
}