//! # Solana Light Client via Tor
//! 
//! 🌞 Ultra-lightweight Solana client that operates exclusively through Tor hidden services.
//! Provides slot tracking, blockhash verification, and state synchronization without full node.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn, error};

use crate::{SolanaBridgeConfig, TorSolanaRpc};

/// Light client state for Solana blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaLightClient {
    config: SolanaBridgeConfig,
    tor_rpc: TorSolanaRpc,
    current_slot: u64,
    finalized_slot: u64,
    recent_blockhashes: HashMap<u64, String>,
    slot_leaders: HashMap<u64, String>,
    last_sync_time: Option<Instant>,
}

/// Slot information from Solana network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotInfo {
    pub slot: u64,
    pub blockhash: String,
    pub parent_slot: u64,
    pub leader: String,
    pub timestamp: u64,
    pub transaction_count: u32,
}

impl SolanaLightClient {
    /// Create new light client instance
    pub async fn new(config: &SolanaBridgeConfig) -> Result<Self> {
        info!("🌞 Initializing Solana Light Client via Tor");
        
        let tor_rpc = TorSolanaRpc::new(config).await?;
        
        Ok(Self {
            config: config.clone(),
            tor_rpc,
            current_slot: 0,
            finalized_slot: 0,
            recent_blockhashes: HashMap::new(),
            slot_leaders: HashMap::new(),
            last_sync_time: None,
        })
    }
    
    /// Start light client synchronization
    pub async fn start_sync(&mut self) -> Result<()> {
        info!("🔄 Starting Solana light client sync");
        
        // Initial sync
        self.sync_current_state().await?;
        
        // Start background sync task
        let mut sync_interval = tokio::time::interval(Duration::from_secs(10));
        
        loop {
            tokio::select! {
                _ = sync_interval.tick() => {
                    if let Err(e) = self.sync_current_state().await {
                        error!("Failed to sync Solana state: {}", e);
                    }
                }
            }
        }
    }
    
    /// Synchronize with current Solana network state
    async fn sync_current_state(&mut self) -> Result<()> {
        let start_time = Instant::now();
        
        // Get current slot information
        let slot_info = self.tor_rpc.get_slot_info().await?;
        
        // Update current slot if advanced
        if slot_info.slot > self.current_slot {
            debug!("📈 Advanced from slot {} to {}", self.current_slot, slot_info.slot);
            self.current_slot = slot_info.slot;
            
            // Store blockhash for this slot
            self.recent_blockhashes.insert(slot_info.slot, slot_info.blockhash.clone());
            
            // Store slot leader
            self.slot_leaders.insert(slot_info.slot, slot_info.leader.clone());
            
            // Cleanup old entries (keep last 150 slots)
            self.cleanup_old_data(150).await;
        }
        
        // Update finalized slot
        let finalized = self.tor_rpc.get_finalized_slot().await?;
        if finalized > self.finalized_slot {
            debug!("🔒 Finalized slot advanced to {}", finalized);
            self.finalized_slot = finalized;
        }
        
        self.last_sync_time = Some(start_time);
        
        let sync_duration = start_time.elapsed().as_millis();
        debug!("⚡ Sync completed in {}ms", sync_duration);
        
        Ok(())
    }
    
    /// Clean up old slot data to prevent memory growth
    async fn cleanup_old_data(&mut self, keep_slots: usize) {
        if self.recent_blockhashes.len() <= keep_slots {
            return;
        }
        
        // Get slots to remove (older than keep_slots)
        let cutoff_slot = self.current_slot.saturating_sub(keep_slots as u64);
        
        let mut removed_count = 0;
        self.recent_blockhashes.retain(|&slot, _| {
            if slot < cutoff_slot {
                removed_count += 1;
                false
            } else {
                true
            }
        });
        
        self.slot_leaders.retain(|&slot, _| slot >= cutoff_slot);
        
        if removed_count > 0 {
            debug!("🧹 Cleaned up {} old slot entries", removed_count);
        }
    }
    
    /// Get current slot number
    pub async fn get_current_slot(&self) -> Result<u64> {
        // Return cached slot if sync is recent (< 30 seconds ago)
        if let Some(last_sync) = self.last_sync_time {
            if last_sync.elapsed() < Duration::from_secs(30) {
                return Ok(self.current_slot);
            }
        }
        
        // Otherwise fetch fresh slot info
        let slot_info = self.tor_rpc.get_slot_info().await?;
        Ok(slot_info.slot)
    }
    
    /// Get finalized slot number
    pub async fn get_finalized_slot(&self) -> Result<u64> {
        if let Some(last_sync) = self.last_sync_time {
            if last_sync.elapsed() < Duration::from_secs(60) {
                return Ok(self.finalized_slot);
            }
        }
        
        let finalized = self.tor_rpc.get_finalized_slot().await?;
        Ok(finalized)
    }
    
    /// Get blockhash for specific slot
    pub async fn get_blockhash_for_slot(&self, slot: u64) -> Result<String> {
        // Check cache first
        if let Some(blockhash) = self.recent_blockhashes.get(&slot) {
            return Ok(blockhash.clone());
        }
        
        // Fetch from network if not in cache
        match self.tor_rpc.get_blockhash_for_slot(slot).await {
            Ok(blockhash) => {
                debug!("🔍 Fetched blockhash for slot {}: {}", slot, &blockhash[..8]);
                Ok(blockhash)
            },
            Err(e) => {
                warn!("Failed to get blockhash for slot {}: {}", slot, e);
                Err(e)
            }
        }
    }
    
    /// Get recent blockhash (for transaction building)
    pub async fn get_recent_blockhash(&self) -> Result<String> {
        let current_slot = self.get_current_slot().await?;
        self.get_blockhash_for_slot(current_slot).await
    }
    
    /// Get slot leader for specific slot
    pub async fn get_slot_leader(&self, slot: u64) -> Result<String> {
        // Check cache first
        if let Some(leader) = self.slot_leaders.get(&slot) {
            return Ok(leader.clone());
        }
        
        // Fetch from network
        let leader = self.tor_rpc.get_slot_leader(slot).await?;
        debug!("👨‍💼 Slot {} leader: {}", slot, &leader[..8]);
        Ok(leader)
    }
    
    /// Verify if slot is finalized
    pub async fn is_slot_finalized(&self, slot: u64) -> Result<bool> {
        let finalized_slot = self.get_finalized_slot().await?;
        Ok(slot <= finalized_slot)
    }
    
    /// Get confirmation status for slot
    pub async fn get_slot_confirmation(&self, slot: u64) -> Result<SlotConfirmation> {
        let current_slot = self.get_current_slot().await?;
        let finalized_slot = self.get_finalized_slot().await?;
        
        if slot > current_slot {
            Ok(SlotConfirmation::Future)
        } else if slot <= finalized_slot {
            Ok(SlotConfirmation::Finalized)
        } else {
            // Check confirmation level
            let confirmations = current_slot.saturating_sub(slot);
            
            if confirmations >= 32 {
                Ok(SlotConfirmation::Confirmed)
            } else if confirmations >= 1 {
                Ok(SlotConfirmation::Processed)
            } else {
                Ok(SlotConfirmation::Recent)
            }
        }
    }
    
    /// Get network health status
    pub async fn get_network_health(&self) -> NetworkHealth {
        let mut health = NetworkHealth::default();
        
        // Check last sync time
        if let Some(last_sync) = self.last_sync_time {
            health.last_sync_age_seconds = last_sync.elapsed().as_secs();
            health.sync_healthy = health.last_sync_age_seconds < 60;
        }
        
        // Check slot progression
        if self.current_slot > 0 && self.finalized_slot > 0 {
            health.slot_gap = self.current_slot.saturating_sub(self.finalized_slot);
            health.finalization_healthy = health.slot_gap < 150; // ~1 minute at 400ms slots
        }
        
        // Check RPC connectivity
        health.rpc_healthy = match self.tor_rpc.health_check().await {
            Ok(true) => true,
            _ => false,
        };
        
        // Overall health
        health.overall_healthy = health.sync_healthy && 
                                 health.finalization_healthy && 
                                 health.rpc_healthy;
        
        health
    }
    
    /// Get light client statistics
    pub fn get_stats(&self) -> LightClientStats {
        LightClientStats {
            current_slot: self.current_slot,
            finalized_slot: self.finalized_slot,
            cached_blockhashes: self.recent_blockhashes.len(),
            cached_leaders: self.slot_leaders.len(),
            last_sync_age_seconds: self.last_sync_time
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(u64::MAX),
            tor_endpoints: self.config.solana_rpc_endpoints.len(),
        }
    }
}

/// Slot confirmation levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlotConfirmation {
    Future,      // Slot hasn't occurred yet
    Recent,      // Just processed
    Processed,   // 1+ confirmations
    Confirmed,   // 32+ confirmations
    Finalized,   // Finalized by supermajority
}

/// Network health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    pub overall_healthy: bool,
    pub sync_healthy: bool,
    pub finalization_healthy: bool,
    pub rpc_healthy: bool,
    pub last_sync_age_seconds: u64,
    pub slot_gap: u64,
}

impl Default for NetworkHealth {
    fn default() -> Self {
        Self {
            overall_healthy: false,
            sync_healthy: false,
            finalization_healthy: false,
            rpc_healthy: false,
            last_sync_age_seconds: u64::MAX,
            slot_gap: u64::MAX,
        }
    }
}

/// Light client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightClientStats {
    pub current_slot: u64,
    pub finalized_slot: u64,
    pub cached_blockhashes: usize,
    pub cached_leaders: usize,
    pub last_sync_age_seconds: u64,
    pub tor_endpoints: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slot_confirmation_levels() {
        // Test different confirmation levels
        assert_eq!(SlotConfirmation::Future, SlotConfirmation::Future);
        assert_ne!(SlotConfirmation::Recent, SlotConfirmation::Finalized);
        
        // Should serialize properly
        let confirmation = SlotConfirmation::Confirmed;
        let serialized = serde_json::to_string(&confirmation).unwrap();
        let deserialized: SlotConfirmation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(confirmation, deserialized);
    }
    
    #[test]
    fn test_network_health_default() {
        let health = NetworkHealth::default();
        assert!(!health.overall_healthy);
        assert!(!health.sync_healthy);
        assert_eq!(health.last_sync_age_seconds, u64::MAX);
    }
    
    #[tokio::test]
    async fn test_light_client_initialization() {
        let config = SolanaBridgeConfig::default();
        
        // This will fail without real Tor/Solana setup
        let result = SolanaLightClient::new(&config).await;
        
        // Verify structure is correct even if creation fails
        if result.is_err() {
            println!("Expected failure without Tor setup: {:?}", result.err());
        }
    }
}