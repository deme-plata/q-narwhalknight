// Q-NarwhalKnight Hierarchical Caching System
// Phase 2: Intelligent caching for 4x TPS improvement (25k → 100k TPS)

pub mod cache_coherency;
pub mod cache_levels;
pub mod hierarchical_cache;
pub mod memory_optimization;
pub mod metrics;
pub mod prefetch_engine;

use anyhow::Result;
use q_types::{Hash256, StateKey, StateValue, Transaction, Vertex, VertexId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main cache configuration for Phase 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 cache size (hot vertex cache)
    pub l1_size_mb: usize,
    /// L2 cache size (block cache)
    pub l2_size_mb: usize,
    /// L3 cache size (state cache)
    pub l3_size_mb: usize,
    /// Enable ML-powered prefetching
    pub ml_prefetch_enabled: bool,
    /// Cache hit ratio target (0.0-1.0)
    pub target_hit_ratio: f64,
    /// Cache coherency timeout
    pub coherency_timeout_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size_mb: 1,    // 1MB L1 hot vertex cache
            l2_size_mb: 100,  // 100MB L2 block cache
            l3_size_mb: 1000, // 1GB L3 state cache
            ml_prefetch_enabled: true,
            target_hit_ratio: 0.9,    // 90% hit ratio target
            coherency_timeout_ms: 10, // 10ms coherency timeout
        }
    }
}

/// Cache levels in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLevel {
    L1, // Hot vertex cache (fastest, smallest)
    L2, // Block cache (medium speed/size)
    L3, // State cache (slowest, largest)
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub key: Hash256,
    pub value: T,
    pub access_count: u32,
    pub last_accessed: std::time::SystemTime,
    pub cache_level: CacheLevel,
    pub size_bytes: usize,
    pub is_dirty: bool,
}

/// Cache hit/miss statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub total_requests: u64,
    pub hit_ratio: f64,
    pub average_access_time_ns: f64,
}

/// Cache operation result
#[derive(Debug, Clone)]
pub enum CacheResult<T> {
    Hit(T, CacheLevel),
    Miss,
    Prefetched(T),
}

/// Main hierarchical cache engine
#[derive(Debug)]
pub struct CacheEngine {
    config: CacheConfig,
    l1_cache: Arc<RwLock<cache_levels::L1Cache>>,
    l2_cache: Arc<RwLock<cache_levels::L2Cache>>,
    l3_cache: Arc<RwLock<cache_levels::L3Cache>>,
    prefetch_engine: Arc<prefetch_engine::PrefetchEngine>,
    coherency_manager: Arc<cache_coherency::CoherencyManager>,
    stats: Arc<RwLock<CacheStats>>,
    metrics: Arc<metrics::CacheMetricsCollector>,
}

impl CacheEngine {
    /// Create new hierarchical cache engine
    pub async fn new(config: CacheConfig) -> Result<Self> {
        tracing::info!(
            "Creating hierarchical cache engine with L1={}MB, L2={}MB, L3={}MB",
            config.l1_size_mb,
            config.l2_size_mb,
            config.l3_size_mb
        );

        let l1_cache = Arc::new(RwLock::new(
            cache_levels::L1Cache::new(config.l1_size_mb).await?,
        ));
        let l2_cache = Arc::new(RwLock::new(
            cache_levels::L2Cache::new(config.l2_size_mb).await?,
        ));
        let l3_cache = Arc::new(RwLock::new(
            cache_levels::L3Cache::new(config.l3_size_mb).await?,
        ));

        let prefetch_engine = Arc::new(prefetch_engine::PrefetchEngine::new(config.clone()).await?);
        let coherency_manager =
            Arc::new(cache_coherency::CoherencyManager::new(config.clone()).await?);
        let stats = Arc::new(RwLock::new(CacheStats::default()));
        let metrics = Arc::new(metrics::CacheMetricsCollector::new());

        Ok(Self {
            config,
            l1_cache,
            l2_cache,
            l3_cache,
            prefetch_engine,
            coherency_manager,
            stats,
            metrics,
        })
    }

    /// Get vertex with hierarchical cache lookup
    pub async fn get_vertex(&self, vertex_id: &VertexId) -> Result<CacheResult<Vertex>> {
        let start_time = std::time::Instant::now();
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        drop(stats);

        let key = vertex_id; // VertexId and Hash256 are both [u8; 32]

        // L1 cache lookup (hot vertices)
        if let Some(vertex) = self.l1_cache.read().await.get(&key).await? {
            let mut stats = self.stats.write().await;
            stats.l1_hits += 1;
            self.update_hit_ratio(&mut stats);

            tracing::debug!("L1 cache hit for vertex {:?}", vertex_id);
            return Ok(CacheResult::Hit(vertex, CacheLevel::L1));
        }

        let mut stats = self.stats.write().await;
        stats.l1_misses += 1;
        drop(stats);

        // L2 cache lookup (blocks)
        if let Some(vertex) = self.l2_cache.read().await.get(&key).await? {
            let mut stats = self.stats.write().await;
            stats.l2_hits += 1;
            self.update_hit_ratio(&mut stats);
            drop(stats);

            // Promote to L1 for faster future access
            self.l1_cache
                .write()
                .await
                .put(*key, vertex.clone())
                .await?;

            tracing::debug!("L2 cache hit for vertex {:?}, promoted to L1", vertex_id);
            return Ok(CacheResult::Hit(vertex, CacheLevel::L2));
        }

        let mut stats = self.stats.write().await;
        stats.l2_misses += 1;
        drop(stats);

        // L3 cache lookup (state data)
        if let Some(vertex) = self.l3_cache.read().await.get(&key).await? {
            let mut stats = self.stats.write().await;
            stats.l3_hits += 1;
            self.update_hit_ratio(&mut stats);
            drop(stats);

            // Promote through cache hierarchy
            self.l2_cache
                .write()
                .await
                .put(*key, vertex.clone())
                .await?;
            self.l1_cache
                .write()
                .await
                .put(*key, vertex.clone())
                .await?;

            tracing::debug!("L3 cache hit for vertex {:?}, promoted to L1/L2", vertex_id);
            return Ok(CacheResult::Hit(vertex, CacheLevel::L3));
        }

        let mut stats = self.stats.write().await;
        stats.l3_misses += 1;
        self.update_hit_ratio(&mut stats);
        drop(stats);

        // Cache miss - trigger prefetching for related data
        if self.config.ml_prefetch_enabled {
            self.prefetch_engine.trigger_prefetch(&key).await?;
        }

        let access_time = start_time.elapsed();
        self.metrics.record_access(access_time, false).await;

        tracing::debug!("Cache miss for vertex {:?}", vertex_id);
        Ok(CacheResult::Miss)
    }

    /// Store vertex in appropriate cache level
    pub async fn put_vertex(&self, vertex_id: &VertexId, vertex: Vertex) -> Result<()> {
        let key = vertex_id; // VertexId and Hash256 are both [u8; 32]
        let start_time = std::time::Instant::now();

        // Determine cache level based on access patterns
        let cache_level = self.prefetch_engine.determine_cache_level(&key).await?;

        match cache_level {
            CacheLevel::L1 => {
                self.l1_cache.write().await.put(*key, vertex).await?;
                tracing::debug!("Stored vertex {:?} in L1 cache", vertex_id);
            }
            CacheLevel::L2 => {
                self.l2_cache.write().await.put(*key, vertex).await?;
                tracing::debug!("Stored vertex {:?} in L2 cache", vertex_id);
            }
            CacheLevel::L3 => {
                self.l3_cache.write().await.put(*key, vertex).await?;
                tracing::debug!("Stored vertex {:?} in L3 cache", vertex_id);
            }
        }

        let access_time = start_time.elapsed();
        self.metrics.record_access(access_time, true).await;

        Ok(())
    }

    /// Get state value with caching
    pub async fn get_state(&self, key: &StateKey) -> Result<CacheResult<StateValue>> {
        // Similar hierarchical lookup for state data
        let hash_key: Hash256 = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(key);
            hasher.finalize().into()
        };

        // Check L3 first for state data (most likely location)
        if let Some(value) = self.l3_cache.read().await.get_state(&hash_key).await? {
            let mut stats = self.stats.write().await;
            stats.l3_hits += 1;
            self.update_hit_ratio(&mut stats);
            return Ok(CacheResult::Hit(value, CacheLevel::L3));
        }

        let mut stats = self.stats.write().await;
        stats.l3_misses += 1;
        self.update_hit_ratio(&mut stats);
        drop(stats);

        Ok(CacheResult::Miss)
    }

    /// Store state value in cache
    pub async fn put_state(&self, key: &StateKey, value: StateValue) -> Result<()> {
        let hash_key: Hash256 = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(key);
            hasher.finalize().into()
        };

        // State data typically goes to L3
        self.l3_cache
            .write()
            .await
            .put_state(hash_key, value)
            .await?;

        tracing::debug!("Stored state key in L3 cache");
        Ok(())
    }

    /// Invalidate cache entry across all levels
    pub async fn invalidate(&self, key: &Hash256) -> Result<()> {
        // Invalidate across all cache levels
        let l1_removed = self.l1_cache.write().await.remove(key).await?;
        let l2_removed = self.l2_cache.write().await.remove(key).await?;
        let l3_removed = self.l3_cache.write().await.remove(key).await?;

        if l1_removed || l2_removed || l3_removed {
            tracing::debug!("Invalidated cache entry for key {:?}", key);
        }

        Ok(())
    }

    /// Flush all dirty entries to persistent storage
    pub async fn flush_dirty(&self) -> Result<u64> {
        let mut total_flushed = 0;

        total_flushed += self.l1_cache.write().await.flush_dirty().await?;
        total_flushed += self.l2_cache.write().await.flush_dirty().await?;
        total_flushed += self.l3_cache.write().await.flush_dirty().await?;

        if total_flushed > 0 {
            tracing::info!("Flushed {} dirty cache entries", total_flushed);
        }

        Ok(total_flushed)
    }

    /// Get current cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Update cache hit ratio calculation
    fn update_hit_ratio(&self, stats: &mut CacheStats) {
        let total_hits = stats.l1_hits + stats.l2_hits + stats.l3_hits;
        if stats.total_requests > 0 {
            stats.hit_ratio = total_hits as f64 / stats.total_requests as f64;
        }
    }

    /// Warm up cache with frequently accessed data
    pub async fn warm_cache(&self, keys: Vec<Hash256>) -> Result<u32> {
        let mut warmed_count = 0;

        for key in keys {
            if let Ok(_) = self.prefetch_engine.prefetch_data(&key).await {
                warmed_count += 1;
            }
        }

        tracing::info!("Warmed cache with {} entries", warmed_count);
        Ok(warmed_count)
    }

    /// Get cache memory usage across all levels
    pub async fn get_memory_usage(&self) -> Result<HashMap<CacheLevel, usize>> {
        let mut usage = HashMap::new();

        usage.insert(
            CacheLevel::L1,
            self.l1_cache.read().await.memory_usage().await?,
        );
        usage.insert(
            CacheLevel::L2,
            self.l2_cache.read().await.memory_usage().await?,
        );
        usage.insert(
            CacheLevel::L3,
            self.l3_cache.read().await.memory_usage().await?,
        );

        Ok(usage)
    }

    /// Optimize cache configuration based on access patterns
    pub async fn optimize_configuration(&mut self) -> Result<CacheConfig> {
        let stats = self.get_stats().await;
        let mut new_config = self.config.clone();

        // Adjust cache sizes based on hit ratios
        if stats.l1_hits as f64 / (stats.l1_hits + stats.l1_misses) as f64 > 0.95 {
            // L1 very effective, consider increasing size
            new_config.l1_size_mb = (self.config.l1_size_mb * 120 / 100).min(10);
            // Max 10MB
        }

        if stats.hit_ratio < self.config.target_hit_ratio {
            // Overall hit ratio low, increase L2/L3 sizes
            new_config.l2_size_mb = (self.config.l2_size_mb * 110 / 100).min(200); // Max 200MB
            new_config.l3_size_mb = (self.config.l3_size_mb * 105 / 100).min(2000);
            // Max 2GB
        }

        tracing::info!(
            "Optimized cache configuration: L1={}MB, L2={}MB, L3={}MB",
            new_config.l1_size_mb,
            new_config.l2_size_mb,
            new_config.l3_size_mb
        );

        self.config = new_config.clone();
        Ok(new_config)
    }
}

/// Phase 2 performance target validation
pub async fn validate_phase2_targets(cache_engine: &CacheEngine) -> Result<bool> {
    let stats = cache_engine.get_stats().await;

    // Phase 2 success criteria
    let hit_ratio_target = 0.90; // 90% hit ratio
    let memory_limit_gb = 2.0; // <2GB memory usage

    let memory_usage = cache_engine.get_memory_usage().await?;
    let total_memory_mb = memory_usage.values().sum::<usize>();
    let total_memory_gb = total_memory_mb as f64 / 1024.0;

    let hit_ratio_ok = stats.hit_ratio >= hit_ratio_target;
    let memory_ok = total_memory_gb <= memory_limit_gb;

    tracing::info!("Phase 2 Target Validation:");
    tracing::info!(
        "  Hit Ratio: {:.1}% (target: {:.1}%) - {}",
        stats.hit_ratio * 100.0,
        hit_ratio_target * 100.0,
        if hit_ratio_ok { "✅" } else { "❌" }
    );
    tracing::info!(
        "  Memory Usage: {:.1}GB (limit: {:.1}GB) - {}",
        total_memory_gb,
        memory_limit_gb,
        if memory_ok { "✅" } else { "❌" }
    );

    Ok(hit_ratio_ok && memory_ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_engine_creation() {
        let config = CacheConfig::default();
        let engine = CacheEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_hierarchical_cache_lookup() {
        let config = CacheConfig::default();
        let engine = CacheEngine::new(config).await.unwrap();

        // Test cache miss
        let vertex_id = [1u8; 32];
        let result = engine.get_vertex(&vertex_id).await.unwrap();

        match result {
            CacheResult::Miss => {
                // Expected for empty cache
            }
            _ => panic!("Expected cache miss"),
        }
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let engine = CacheEngine::new(config).await.unwrap();

        let stats = engine.get_stats().await;
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.hit_ratio, 0.0);
    }
}
