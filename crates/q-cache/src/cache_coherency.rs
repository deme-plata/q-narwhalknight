//! Cache Coherency Manager for Multi-Shard Consistency
//!
//! Ensures cache consistency across multiple shards in the
//! Q-NarwhalKnight distributed system.

use crate::{CacheConfig, CacheLevel, Hash256};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Custom serde module for SystemTime
mod systemtime_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        duration.as_nanos().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u128::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_nanos(nanos as u64))
    }
}

/// Cache coherency protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherencyMessage {
    /// Invalidate cache entry across all shards
    InvalidateEntry {
        key: Hash256,
        shard_id: u32,
        #[serde(with = "systemtime_serde")]
        timestamp: SystemTime,
    },
    /// Notify of cache update
    CacheUpdate {
        key: Hash256,
        shard_id: u32,
        cache_level: CacheLevel,
        #[serde(with = "systemtime_serde")]
        timestamp: SystemTime,
    },
    /// Request cache entry from other shards
    CacheRequest {
        key: Hash256,
        requesting_shard: u32,
        #[serde(with = "systemtime_serde")]
        timestamp: SystemTime,
    },
    /// Response to cache request
    CacheResponse {
        key: Hash256,
        data_hash: Option<Hash256>, // None if not found
        cache_level: Option<CacheLevel>,
        responding_shard: u32,
        #[serde(with = "systemtime_serde")]
        timestamp: SystemTime,
    },
    /// Heartbeat to maintain coherency session
    Heartbeat {
        shard_id: u32,
        #[serde(with = "systemtime_serde")]
        timestamp: SystemTime,
    },
}

/// Cache entry state for coherency tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEntryState {
    /// Entry is valid and up-to-date
    Valid,
    /// Entry has been modified locally
    Modified,
    /// Entry has been invalidated
    Invalid,
    /// Entry is shared across multiple shards
    Shared,
    /// Entry is exclusively owned by one shard
    Exclusive,
}

/// Coherency entry metadata
#[derive(Debug, Clone)]
pub struct CoherencyEntry {
    pub key: Hash256,
    pub state: CacheEntryState,
    pub owner_shard: Option<u32>,
    pub shared_shards: HashSet<u32>,
    pub last_modified: SystemTime,
    pub invalidation_pending: bool,
}

/// Coherency statistics
#[derive(Debug, Clone, Default)]
pub struct CoherencyMetrics {
    pub invalidations_sent: u64,
    pub invalidations_received: u64,
    pub cache_requests_sent: u64,
    pub cache_requests_received: u64,
    pub coherency_violations: u64,
    pub average_invalidation_latency_ms: f64,
    pub active_shards: u32,
}

/// Cache coherency manager
#[derive(Debug)]
pub struct CoherencyManager {
    config: CacheConfig,
    local_shard_id: u32,
    coherency_table: Arc<RwLock<HashMap<Hash256, CoherencyEntry>>>,
    peer_shards: Arc<RwLock<HashSet<u32>>>,
    message_sender: Arc<RwLock<Option<mpsc::UnboundedSender<CoherencyMessage>>>>,
    metrics: Arc<RwLock<CoherencyMetrics>>,
    pending_invalidations: Arc<RwLock<HashMap<Hash256, Instant>>>,
}

impl CoherencyManager {
    /// Create new cache coherency manager
    pub async fn new(config: CacheConfig) -> Result<Self> {
        info!("🔗 Initializing cache coherency manager");

        Ok(Self {
            config,
            local_shard_id: 0, // Will be set during initialization
            coherency_table: Arc::new(RwLock::new(HashMap::new())),
            peer_shards: Arc::new(RwLock::new(HashSet::new())),
            message_sender: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(CoherencyMetrics::default())),
            pending_invalidations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize coherency manager with shard information
    pub async fn initialize(&mut self, local_shard_id: u32, peer_shards: Vec<u32>) -> Result<()> {
        self.local_shard_id = local_shard_id;

        let mut peers = self.peer_shards.write().await;
        for shard_id in peer_shards {
            peers.insert(shard_id);
        }
        let peer_count = peers.len();
        drop(peers);

        let mut metrics = self.metrics.write().await;
        metrics.active_shards = peer_count as u32 + 1; // +1 for local shard

        info!(
            "✅ Cache coherency initialized for shard {} with {} peers",
            local_shard_id, peer_count
        );

        Ok(())
    }

    /// Set message sender for cross-shard communication
    pub async fn set_message_sender(
        &self,
        sender: mpsc::UnboundedSender<CoherencyMessage>,
    ) -> Result<()> {
        let mut msg_sender = self.message_sender.write().await;
        *msg_sender = Some(sender);
        Ok(())
    }

    /// Record cache entry creation/update
    pub async fn on_cache_update(&self, key: Hash256, cache_level: CacheLevel) -> Result<()> {
        let timestamp = SystemTime::now();

        // Update local coherency table
        {
            let mut table = self.coherency_table.write().await;
            let entry = table.entry(key).or_insert_with(|| CoherencyEntry {
                key,
                state: CacheEntryState::Exclusive,
                owner_shard: Some(self.local_shard_id),
                shared_shards: HashSet::new(),
                last_modified: timestamp,
                invalidation_pending: false,
            });

            entry.state = CacheEntryState::Modified;
            entry.last_modified = timestamp;
            entry.owner_shard = Some(self.local_shard_id);
        }

        // Notify peer shards of cache update
        self.broadcast_cache_update(key, cache_level, timestamp)
            .await?;

        debug!(
            "📝 Recorded cache update for key {:?} at level {:?}",
            key, cache_level
        );
        Ok(())
    }

    /// Handle cache entry access (read)
    pub async fn on_cache_access(&self, key: Hash256) -> Result<bool> {
        let table = self.coherency_table.read().await;

        if let Some(entry) = table.get(&key) {
            match entry.state {
                CacheEntryState::Valid | CacheEntryState::Shared | CacheEntryState::Exclusive => {
                    debug!("✅ Cache access allowed for key {:?}", key);
                    return Ok(true);
                }
                CacheEntryState::Invalid => {
                    debug!("❌ Cache access denied - entry invalid for key {:?}", key);
                    return Ok(false);
                }
                CacheEntryState::Modified => {
                    // Check if we're the owner
                    if entry.owner_shard == Some(self.local_shard_id) {
                        debug!(
                            "✅ Cache access allowed - local ownership for key {:?}",
                            key
                        );
                        return Ok(true);
                    } else {
                        debug!(
                            "❌ Cache access denied - modified by other shard for key {:?}",
                            key
                        );
                        return Ok(false);
                    }
                }
            }
        }

        // Entry not found in coherency table - request from peers
        self.request_cache_entry(key).await?;
        Ok(false) // Access denied until we get response
    }

    /// Invalidate cache entry across all shards
    pub async fn invalidate_entry(&self, key: Hash256) -> Result<()> {
        let timestamp = SystemTime::now();
        let start_time = Instant::now();

        // Update local coherency table
        {
            let mut table = self.coherency_table.write().await;
            if let Some(entry) = table.get_mut(&key) {
                entry.state = CacheEntryState::Invalid;
                entry.last_modified = timestamp;
                entry.invalidation_pending = true;
            }
        }

        // Record pending invalidation for latency tracking
        {
            let mut pending = self.pending_invalidations.write().await;
            pending.insert(key, start_time);
        }

        // Broadcast invalidation to peer shards
        self.broadcast_invalidation(key, timestamp).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.invalidations_sent += 1;
        }

        info!("🔄 Initiated cache invalidation for key {:?}", key);
        Ok(())
    }

    /// Process incoming coherency message
    pub async fn handle_coherency_message(&self, message: CoherencyMessage) -> Result<()> {
        match message {
            CoherencyMessage::InvalidateEntry {
                key,
                shard_id,
                timestamp,
            } => {
                self.handle_invalidation(key, shard_id, timestamp).await?;
            }
            CoherencyMessage::CacheUpdate {
                key,
                shard_id,
                cache_level,
                timestamp,
            } => {
                self.handle_cache_update(key, shard_id, cache_level, timestamp)
                    .await?;
            }
            CoherencyMessage::CacheRequest {
                key,
                requesting_shard,
                timestamp,
            } => {
                self.handle_cache_request(key, requesting_shard, timestamp)
                    .await?;
            }
            CoherencyMessage::CacheResponse {
                key,
                data_hash,
                cache_level,
                responding_shard,
                timestamp,
            } => {
                self.handle_cache_response(
                    key,
                    data_hash,
                    cache_level,
                    responding_shard,
                    timestamp,
                )
                .await?;
            }
            CoherencyMessage::Heartbeat {
                shard_id,
                timestamp,
            } => {
                self.handle_heartbeat(shard_id, timestamp).await?;
            }
        }

        Ok(())
    }

    /// Handle invalidation message from peer shard
    async fn handle_invalidation(
        &self,
        key: Hash256,
        shard_id: u32,
        timestamp: SystemTime,
    ) -> Result<()> {
        debug!(
            "📨 Received invalidation for key {:?} from shard {}",
            key, shard_id
        );

        // Update local coherency table
        {
            let mut table = self.coherency_table.write().await;
            if let Some(entry) = table.get_mut(&key) {
                if entry.last_modified <= timestamp {
                    entry.state = CacheEntryState::Invalid;
                    entry.last_modified = timestamp;

                    // Remove from shared shards if we were sharing
                    entry.shared_shards.remove(&self.local_shard_id);
                }
            } else {
                // Create new entry as invalid
                let entry = CoherencyEntry {
                    key,
                    state: CacheEntryState::Invalid,
                    owner_shard: Some(shard_id),
                    shared_shards: HashSet::new(),
                    last_modified: timestamp,
                    invalidation_pending: false,
                };
                table.insert(key, entry);
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.invalidations_received += 1;
        }

        // This would integrate with the actual cache to remove the entry
        self.perform_local_invalidation(key).await?;

        debug!("✅ Processed invalidation for key {:?}", key);
        Ok(())
    }

    /// Handle cache update message from peer shard
    async fn handle_cache_update(
        &self,
        key: Hash256,
        shard_id: u32,
        cache_level: CacheLevel,
        timestamp: SystemTime,
    ) -> Result<()> {
        debug!(
            "📨 Received cache update for key {:?} from shard {} at level {:?}",
            key, shard_id, cache_level
        );

        // Update coherency table
        {
            let mut table = self.coherency_table.write().await;
            let entry = table.entry(key).or_insert_with(|| CoherencyEntry {
                key,
                state: CacheEntryState::Shared,
                owner_shard: None,
                shared_shards: HashSet::new(),
                last_modified: timestamp,
                invalidation_pending: false,
            });

            if entry.last_modified <= timestamp {
                entry.shared_shards.insert(shard_id);
                entry.last_modified = timestamp;

                // If this entry was exclusively ours, make it shared
                if entry.owner_shard == Some(self.local_shard_id) {
                    entry.state = CacheEntryState::Shared;
                    entry.owner_shard = None;
                }
            }
        }

        Ok(())
    }

    /// Request cache entry from peer shards
    async fn request_cache_entry(&self, key: Hash256) -> Result<()> {
        let timestamp = SystemTime::now();
        let message = CoherencyMessage::CacheRequest {
            key,
            requesting_shard: self.local_shard_id,
            timestamp,
        };

        self.send_to_all_peers(message).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.cache_requests_sent += 1;
        }

        debug!(
            "📤 Requested cache entry for key {:?} from peer shards",
            key
        );
        Ok(())
    }

    /// Handle cache request from peer shard
    async fn handle_cache_request(
        &self,
        key: Hash256,
        requesting_shard: u32,
        timestamp: SystemTime,
    ) -> Result<()> {
        debug!(
            "📨 Received cache request for key {:?} from shard {}",
            key, requesting_shard
        );

        // Check if we have the entry and can share it
        let (data_hash, cache_level) = {
            let table = self.coherency_table.read().await;
            if let Some(entry) = table.get(&key) {
                match entry.state {
                    CacheEntryState::Valid
                    | CacheEntryState::Shared
                    | CacheEntryState::Exclusive => {
                        // We can share this entry
                        (Some(key), Some(CacheLevel::L3)) // Simplified - would be actual data hash and level
                    }
                    _ => (None, None),
                }
            } else {
                (None, None)
            }
        };

        // Send response
        let response = CoherencyMessage::CacheResponse {
            key,
            data_hash,
            cache_level,
            responding_shard: self.local_shard_id,
            timestamp: SystemTime::now(),
        };

        self.send_to_shard(requesting_shard, response).await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.cache_requests_received += 1;
        }

        Ok(())
    }

    /// Handle cache response from peer shard
    async fn handle_cache_response(
        &self,
        key: Hash256,
        data_hash: Option<Hash256>,
        cache_level: Option<CacheLevel>,
        responding_shard: u32,
        timestamp: SystemTime,
    ) -> Result<()> {
        debug!(
            "📨 Received cache response for key {:?} from shard {}",
            key, responding_shard
        );

        if let (Some(_hash), Some(level)) = (data_hash, cache_level) {
            // Update coherency table to reflect shared state
            let mut table = self.coherency_table.write().await;
            let entry = table.entry(key).or_insert_with(|| CoherencyEntry {
                key,
                state: CacheEntryState::Shared,
                owner_shard: None,
                shared_shards: HashSet::new(),
                last_modified: timestamp,
                invalidation_pending: false,
            });

            entry.shared_shards.insert(responding_shard);
            entry.shared_shards.insert(self.local_shard_id);
            entry.state = CacheEntryState::Shared;

            debug!(
                "✅ Updated coherency table for shared cache entry {:?} at level {:?}",
                key, level
            );
        }

        Ok(())
    }

    /// Handle heartbeat from peer shard
    async fn handle_heartbeat(&self, shard_id: u32, _timestamp: SystemTime) -> Result<()> {
        // Update peer shard status (mark as active)
        let mut peers = self.peer_shards.write().await;
        peers.insert(shard_id);

        debug!("💓 Received heartbeat from shard {}", shard_id);
        Ok(())
    }

    /// Broadcast invalidation to all peer shards
    async fn broadcast_invalidation(&self, key: Hash256, timestamp: SystemTime) -> Result<()> {
        let message = CoherencyMessage::InvalidateEntry {
            key,
            shard_id: self.local_shard_id,
            timestamp,
        };

        self.send_to_all_peers(message).await?;
        Ok(())
    }

    /// Broadcast cache update to all peer shards
    async fn broadcast_cache_update(
        &self,
        key: Hash256,
        cache_level: CacheLevel,
        timestamp: SystemTime,
    ) -> Result<()> {
        let message = CoherencyMessage::CacheUpdate {
            key,
            shard_id: self.local_shard_id,
            cache_level,
            timestamp,
        };

        self.send_to_all_peers(message).await?;
        Ok(())
    }

    /// Send message to all peer shards
    async fn send_to_all_peers(&self, message: CoherencyMessage) -> Result<()> {
        let sender = self.message_sender.read().await;
        if let Some(ref sender) = *sender {
            sender
                .send(message)
                .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;
        } else {
            warn!("⚠️ No message sender configured for coherency manager");
        }
        Ok(())
    }

    /// Send message to specific shard
    async fn send_to_shard(&self, _shard_id: u32, message: CoherencyMessage) -> Result<()> {
        // In a real implementation, this would route to specific shard
        // For now, broadcast to all
        self.send_to_all_peers(message).await
    }

    /// Perform local cache invalidation
    async fn perform_local_invalidation(&self, key: Hash256) -> Result<()> {
        // This would integrate with the actual cache layers to remove entries
        debug!("🗑️ Performing local invalidation for key {:?}", key);

        // Update invalidation timing
        {
            let mut pending = self.pending_invalidations.write().await;
            if let Some(start_time) = pending.remove(&key) {
                let latency = start_time.elapsed().as_millis() as f64;

                let mut metrics = self.metrics.write().await;
                let total_invalidations =
                    metrics.invalidations_sent + metrics.invalidations_received;
                if total_invalidations > 0 {
                    metrics.average_invalidation_latency_ms = (metrics
                        .average_invalidation_latency_ms
                        * (total_invalidations - 1) as f64
                        + latency)
                        / total_invalidations as f64;
                } else {
                    metrics.average_invalidation_latency_ms = latency;
                }
            }
        }

        Ok(())
    }

    /// Get coherency metrics
    pub async fn get_metrics(&self) -> CoherencyMetrics {
        self.metrics.read().await.clone()
    }

    /// Validate coherency performance (<10ms coherency timeout target)
    pub async fn validate_coherency_performance(&self) -> Result<bool> {
        let metrics = self.get_metrics().await;
        let target_latency_ms = self.config.coherency_timeout_ms as f64;

        let latency_ok = metrics.average_invalidation_latency_ms <= target_latency_ms;
        let violations_low =
            metrics.coherency_violations < (metrics.invalidations_received / 100).max(1); // <1% violation rate

        info!("🎯 Cache Coherency Performance Validation:");
        info!(
            "  Average invalidation latency: {:.1}ms (target: {:.0}ms) - {}",
            metrics.average_invalidation_latency_ms,
            target_latency_ms,
            if latency_ok { "✅" } else { "❌" }
        );
        info!(
            "  Coherency violations: {} ({:.2}%) - {}",
            metrics.coherency_violations,
            metrics.coherency_violations as f64 / metrics.invalidations_received.max(1) as f64
                * 100.0,
            if violations_low { "✅" } else { "❌" }
        );
        info!("  Active shards: {}", metrics.active_shards);

        Ok(latency_ok && violations_low)
    }

    /// Periodic maintenance for coherency table cleanup
    pub async fn maintenance_cleanup(&self) -> Result<u32> {
        let now = SystemTime::now();
        let cleanup_threshold = Duration::from_secs(3600); // 1 hour
        let mut cleaned_entries = 0;

        {
            let mut table = self.coherency_table.write().await;
            table.retain(|_, entry| {
                let should_retain = match now.duration_since(entry.last_modified) {
                    Ok(age) => age < cleanup_threshold || entry.state != CacheEntryState::Invalid,
                    Err(_) => true, // Keep if time calculation fails
                };

                if !should_retain {
                    cleaned_entries += 1;
                }

                should_retain
            });
        }

        if cleaned_entries > 0 {
            debug!("🧹 Cleaned {} stale coherency entries", cleaned_entries);
        }

        Ok(cleaned_entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coherency_manager_creation() {
        let config = CacheConfig::default();
        let manager = CoherencyManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_cache_update_tracking() {
        let config = CacheConfig::default();
        let mut manager = CoherencyManager::new(config).await.unwrap();
        manager.initialize(1, vec![2, 3]).await.unwrap();

        let key = [1u8; 32];
        let result = manager.on_cache_update(key, CacheLevel::L1).await;
        assert!(result.is_ok());

        let access_allowed = manager.on_cache_access(key).await.unwrap();
        assert!(access_allowed);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = CacheConfig::default();
        let manager = CoherencyManager::new(config).await.unwrap();

        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.invalidations_sent, 0);
        assert_eq!(metrics.coherency_violations, 0);
    }
}
