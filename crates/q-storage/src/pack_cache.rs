/// Pack Caching for Turbo Sync
///
/// Phase 3: Network Optimization - Server-Side Pack Caching (Week 3-4)
/// Target: 19.2 → 25+ blocks/s (+30% improvement)
///
/// Architecture:
/// - LRU (Least Recently Used) cache for compressed block packs
/// - Reorg safety: Invalidate cache when blockchain reorgs
/// - Cache key: (start_height, end_height, compression_level)
/// - Memory limit: 500 MB default (configurable)
/// - Hit rate target: 80%+ for popular ranges
///
/// Key Innovation:
/// Server-side compression is expensive (40-60ms @ level 1):
///   Without cache: Compress every time → 60ms per request
///   With cache:    Lookup + serve → 2ms per request (30x faster!)
///
/// This dramatically reduces server CPU usage and response time.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::turbo_sync::BlockPack;

/// Cache key for block packs
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct PackCacheKey {
    /// Starting block height
    pub start_height: u64,

    /// Ending block height (inclusive)
    pub end_height: u64,

    /// Compression level used
    pub compression_level: i32,
}

impl PackCacheKey {
    pub fn new(start_height: u64, end_height: u64, compression_level: i32) -> Self {
        Self {
            start_height,
            end_height,
            compression_level,
        }
    }

    /// Get size of the range
    pub fn range_size(&self) -> u64 {
        self.end_height - self.start_height + 1
    }
}

/// Cached pack entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached block pack
    pack: BlockPack,

    /// When this entry was created
    created_at: Instant,

    /// Last time this entry was accessed
    last_accessed: Instant,

    /// Number of times this entry has been accessed
    access_count: u64,

    /// Size in bytes (for memory limit tracking)
    size_bytes: usize,
}

impl CacheEntry {
    fn new(pack: BlockPack) -> Self {
        let size_bytes = pack.compressed_data.len();
        let now = Instant::now();

        Self {
            pack,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes,
        }
    }

    /// Mark this entry as accessed (updates LRU metadata)
    fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Configuration for pack cache
#[derive(Clone, Debug)]
pub struct PackCacheConfig {
    /// Maximum cache size in bytes (default: 500 MB)
    pub max_size_bytes: usize,

    /// Maximum number of entries (default: 1000)
    pub max_entries: usize,

    /// Time-to-live for cache entries (default: 1 hour)
    pub entry_ttl: Duration,

    /// Enable cache (can be disabled for testing)
    pub enabled: bool,

    /// Reorg detection window (invalidate cache if tip changes by this much)
    pub reorg_detection_window: u64,
}

impl Default for PackCacheConfig {
    fn default() -> Self {
        // v6.0.4: RAM-aware pack cache sizing to prevent OOM on small nodes
        let total_ram_mb = {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_memory();
            (sys.total_memory() / (1024 * 1024)) as usize
        };

        let (cache_mb, max_entries) = match total_ram_mb {
            0..=3999     => (64, 200),     // micro: 64MB, 200 entries
            4000..=7999  => (128, 500),    // small (Gamma): 128MB, 500 entries
            8000..=15999 => (256, 1000),   // medium: 256MB
            _            => (500, 1000),   // large+: 500MB (original default)
        };

        Self {
            max_size_bytes: cache_mb * 1024 * 1024,
            max_entries,
            entry_ttl: Duration::from_secs(3600),  // 1 hour
            enabled: true,
            reorg_detection_window: 10,  // Invalidate if tip changes by >10 blocks
        }
    }
}

/// Pack cache statistics
#[derive(Debug, Clone, Default)]
pub struct PackCacheStats {
    /// Total cache hits
    pub hits: u64,

    /// Total cache misses
    pub misses: u64,

    /// Total evictions (LRU)
    pub evictions: u64,

    /// Total invalidations (reorg)
    pub invalidations: u64,

    /// Current cache size in bytes
    pub current_size_bytes: usize,

    /// Current number of entries
    pub current_entries: usize,
}

impl PackCacheStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate average entry size
    pub fn avg_entry_size(&self) -> usize {
        if self.current_entries == 0 {
            0
        } else {
            self.current_size_bytes / self.current_entries
        }
    }
}

/// LRU Pack Cache for TurboSync
pub struct PackCache {
    config: PackCacheConfig,

    /// Cache entries (key -> entry)
    cache: Arc<RwLock<HashMap<PackCacheKey, CacheEntry>>>,

    /// Statistics
    stats: Arc<RwLock<PackCacheStats>>,

    /// Last known blockchain tip height (for reorg detection)
    last_tip_height: Arc<RwLock<u64>>,
}

impl PackCache {
    pub fn new(config: PackCacheConfig) -> Self {
        info!("📦 [PACK CACHE] Initializing with max_size={} MB, max_entries={}",
              config.max_size_bytes / (1024 * 1024),
              config.max_entries);

        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PackCacheStats::default())),
            last_tip_height: Arc::new(RwLock::new(0)),
        }
    }

    /// Get a cached pack (returns None on cache miss)
    pub async fn get(&self, key: &PackCacheKey) -> Option<BlockPack> {
        if !self.config.enabled {
            return None;
        }

        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry has expired
            if entry.created_at.elapsed() > self.config.entry_ttl {
                debug!("📦 [PACK CACHE] Entry expired: {}..{}", key.start_height, key.end_height);
                cache.remove(key);

                let mut stats = self.stats.write().await;
                stats.misses += 1;
                stats.current_entries = cache.len();
                stats.current_size_bytes = cache.values().map(|e| e.size_bytes).sum();

                return None;
            }

            // Cache hit! Update LRU metadata
            entry.access();

            let mut stats = self.stats.write().await;
            stats.hits += 1;

            debug!("✅ [PACK CACHE HIT] {}..{} (access_count: {}, age: {:?})",
                  key.start_height, key.end_height, entry.access_count, entry.created_at.elapsed());

            Some(entry.pack.clone())
        } else {
            // Cache miss
            let mut stats = self.stats.write().await;
            stats.misses += 1;

            debug!("❌ [PACK CACHE MISS] {}..{}", key.start_height, key.end_height);

            None
        }
    }

    /// Put a pack into the cache
    pub async fn put(&self, key: PackCacheKey, pack: BlockPack) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let entry = CacheEntry::new(pack);
        let entry_size = entry.size_bytes;

        let mut cache = self.cache.write().await;

        // Check if we need to evict before inserting
        while cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache).await;
        }

        // Evict entries until we have enough space
        let mut current_size: usize = cache.values().map(|e| e.size_bytes).sum();
        while current_size + entry_size > self.config.max_size_bytes && !cache.is_empty() {
            self.evict_lru(&mut cache).await;
            current_size = cache.values().map(|e| e.size_bytes).sum();
        }

        // Insert new entry
        cache.insert(key.clone(), entry);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.current_entries = cache.len();
        stats.current_size_bytes = cache.values().map(|e| e.size_bytes).sum();

        debug!("📦 [PACK CACHE PUT] {}..{} ({} bytes, total: {}/{} entries, {:.1}/{} MB)",
              key.start_height, key.end_height,
              entry_size,
              stats.current_entries, self.config.max_entries,
              stats.current_size_bytes as f64 / (1024.0 * 1024.0),
              self.config.max_size_bytes / (1024 * 1024));

        Ok(())
    }

    /// Evict least recently used entry
    async fn evict_lru(&self, cache: &mut HashMap<PackCacheKey, CacheEntry>) {
        if let Some((lru_key, _)) = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
        {
            let lru_key = lru_key.clone();
            cache.remove(&lru_key);

            let mut stats = self.stats.write().await;
            stats.evictions += 1;

            debug!("🗑️  [PACK CACHE EVICT] {}..{} (LRU)",
                  lru_key.start_height, lru_key.end_height);
        }
    }

    /// Update blockchain tip height (for reorg detection)
    pub async fn update_tip_height(&self, new_tip: u64) -> Result<()> {
        let mut last_tip = self.last_tip_height.write().await;

        // Detect potential reorg (tip moved backwards or jumped significantly)
        if new_tip < *last_tip || new_tip > *last_tip + self.config.reorg_detection_window {
            warn!("🔄 [PACK CACHE] Potential reorg detected: {} → {} (invalidating cache)",
                  *last_tip, new_tip);

            self.invalidate_all().await?;
        }

        *last_tip = new_tip;
        Ok(())
    }

    /// Invalidate cache entries that might be affected by new blocks
    pub async fn invalidate_recent(&self, cutoff_height: u64) -> Result<()> {
        let mut cache = self.cache.write().await;
        let initial_size = cache.len();

        // Remove all entries that overlap with heights >= cutoff_height
        cache.retain(|key, _| key.end_height < cutoff_height);

        let removed = initial_size - cache.len();

        if removed > 0 {
            let mut stats = self.stats.write().await;
            stats.invalidations += removed as u64;
            stats.current_entries = cache.len();
            stats.current_size_bytes = cache.values().map(|e| e.size_bytes).sum();

            info!("🔄 [PACK CACHE] Invalidated {} entries (cutoff: {})",
                  removed, cutoff_height);
        }

        Ok(())
    }

    /// Invalidate all cache entries (used for reorgs)
    pub async fn invalidate_all(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let removed = cache.len();

        cache.clear();

        if removed > 0 {
            let mut stats = self.stats.write().await;
            stats.invalidations += removed as u64;
            stats.current_entries = 0;
            stats.current_size_bytes = 0;

            warn!("🔄 [PACK CACHE] Invalidated ALL {} entries (full cache clear)",
                  removed);
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> PackCacheStats {
        self.stats.read().await.clone()
    }

    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        stats.hit_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_basic() {
        let config = PackCacheConfig {
            max_size_bytes: 1024 * 1024,  // 1 MB
            max_entries: 10,
            ..Default::default()
        };

        let cache = PackCache::new(config);
        let key = PackCacheKey::new(1, 100, 1);

        // Create a test pack
        let pack = BlockPack {
            start_height: 1,
            end_height: 100,
            compressed_data: vec![0u8; 1024],
            checksum: [0u8; 32],
            compression_ratio: 0.5,
            block_count: 100,
            uncompressed_size: 2048,
            request_id: None,
        };

        // Cache miss initially
        assert!(cache.get(&key).await.is_none());

        // Put into cache
        cache.put(key.clone(), pack.clone()).await.unwrap();

        // Cache hit
        assert!(cache.get(&key).await.is_some());

        // Verify stats
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.current_entries, 1);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let config = PackCacheConfig {
            max_size_bytes: 1024 * 1024,
            max_entries: 2,  // Only allow 2 entries
            ..Default::default()
        };

        let cache = PackCache::new(config);

        // Create test packs
        let pack1 = BlockPack {
            start_height: 1,
            end_height: 100,
            compressed_data: vec![0u8; 1024],
            checksum: [0u8; 32],
            compression_ratio: 0.5,
            block_count: 100,
            uncompressed_size: 2048,
            request_id: None,
        };

        let pack2 = pack1.clone();
        let pack3 = pack1.clone();

        // Insert 3 entries (should trigger eviction)
        cache.put(PackCacheKey::new(1, 100, 1), pack1).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        cache.put(PackCacheKey::new(101, 200, 1), pack2).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        cache.put(PackCacheKey::new(201, 300, 1), pack3).await.unwrap();

        // Verify only 2 entries remain
        let stats = cache.stats().await;
        assert_eq!(stats.current_entries, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_reorg_detection() {
        let config = PackCacheConfig {
            reorg_detection_window: 5,
            ..Default::default()
        };

        let cache = PackCache::new(config);

        // Set initial tip
        cache.update_tip_height(100).await.unwrap();

        // Add some entries
        let pack = BlockPack {
            start_height: 90,
            end_height: 100,
            compressed_data: vec![0u8; 1024],
            checksum: [0u8; 32],
            compression_ratio: 0.5,
            block_count: 11,
            uncompressed_size: 2048,
            request_id: None,
        };

        cache.put(PackCacheKey::new(90, 100, 1), pack).await.unwrap();
        assert_eq!(cache.stats().await.current_entries, 1);

        // Simulate reorg (tip goes backwards)
        cache.update_tip_height(95).await.unwrap();

        // Cache should be invalidated
        assert_eq!(cache.stats().await.current_entries, 0);
        assert_eq!(cache.stats().await.invalidations, 1);
    }
}
