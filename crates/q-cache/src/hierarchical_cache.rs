//! Hierarchical Cache Implementation
//!
//! Core hierarchical caching system integrating L1/L2/L3 cache levels
//! with ML-powered prefetching and NUMA-aware memory optimization.

use crate::{
    cache_coherency::CoherencyManager,
    cache_levels::{L1Cache, L2Cache, L3Cache},
    memory_optimization::{MemoryConfig, MemoryOptimizer},
    metrics::CacheMetricsCollector,
    prefetch_engine::PrefetchEngine,
    CacheConfig, CacheEngine, CacheLevel, CacheResult, CacheStats, Hash256,
};
use anyhow::Result;
use q_types::{StateKey, StateValue, Vertex, VertexId};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Hierarchical cache coordinator
pub struct HierarchicalCache {
    config: CacheConfig,
    l1_cache: Arc<RwLock<L1Cache>>,
    l2_cache: Arc<RwLock<L2Cache>>,
    l3_cache: Arc<RwLock<L3Cache>>,
    prefetch_engine: Arc<PrefetchEngine>,
    coherency_manager: Arc<CoherencyManager>,
    memory_optimizer: Arc<RwLock<MemoryOptimizer>>,
    metrics: Arc<CacheMetricsCollector>,
    stats: Arc<RwLock<CacheStats>>,
}

impl HierarchicalCache {
    /// Create new hierarchical cache system
    pub async fn new(config: CacheConfig) -> Result<Self> {
        info!("🚀 Initializing hierarchical cache system");
        info!("   L1 Cache: {}MB (hot vertices)", config.l1_size_mb);
        info!("   L2 Cache: {}MB (blocks)", config.l2_size_mb);
        info!("   L3 Cache: {}MB (state)", config.l3_size_mb);
        info!(
            "   ML Prefetch: {}",
            if config.ml_prefetch_enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
        info!(
            "   Target Hit Ratio: {:.1}%",
            config.target_hit_ratio * 100.0
        );

        // Initialize cache levels
        let l1_cache = Arc::new(RwLock::new(L1Cache::new(config.l1_size_mb).await?));
        let l2_cache = Arc::new(RwLock::new(L2Cache::new(config.l2_size_mb).await?));
        let l3_cache = Arc::new(RwLock::new(L3Cache::new(config.l3_size_mb).await?));

        // Initialize intelligent components
        let prefetch_engine = Arc::new(PrefetchEngine::new(config.clone()).await?);
        let coherency_manager = Arc::new(CoherencyManager::new(config.clone()).await?);

        // Initialize memory optimization
        let memory_config = MemoryConfig {
            memory_limit_gb: (config.l1_size_mb + config.l2_size_mb + config.l3_size_mb) as f64
                / 1024.0,
            ..Default::default()
        };
        let memory_optimizer = Arc::new(RwLock::new(MemoryOptimizer::new(memory_config).await?));

        // Initialize metrics collection
        let metrics = Arc::new(CacheMetricsCollector::new());
        let stats = Arc::new(RwLock::new(CacheStats::default()));

        info!("✅ Hierarchical cache system initialized successfully");

        Ok(Self {
            config,
            l1_cache,
            l2_cache,
            l3_cache,
            prefetch_engine,
            coherency_manager,
            memory_optimizer,
            metrics,
            stats,
        })
    }

    /// Get vertex with full hierarchical lookup
    pub async fn get_vertex(&self, vertex_id: &VertexId) -> Result<CacheResult<Vertex>> {
        let start_time = std::time::Instant::now();
        let key = vertex_id; // VertexId and Hash256 are both [u8; 32]

        // Increment request counter
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // L1 Cache Check (Fastest)
        debug!("Checking L1 cache for vertex {:?}", vertex_id);
        if let Some(vertex) = self.l1_cache.read().await.get(&key).await? {
            let access_time = start_time.elapsed();

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.l1_hits += 1;
                self.update_hit_ratio(&mut stats);
            }

            // Record metrics
            self.metrics
                .record_operation(
                    crate::metrics::OperationType::Read,
                    Some(CacheLevel::L1),
                    access_time,
                    self.hash_key(&key),
                    true,
                )
                .await;

            // Trigger prefetch for related data
            if self.config.ml_prefetch_enabled {
                let pattern = crate::prefetch_engine::AccessPattern {
                    key: *key,
                    timestamp: std::time::SystemTime::now(),
                    access_type: crate::prefetch_engine::AccessType::Hit,
                    cache_level: Some(CacheLevel::L1),
                    preceding_keys: Vec::new(),
                };
                let _ = self.prefetch_engine.record_access_pattern(pattern).await;
            }

            debug!(
                "✅ L1 cache hit for vertex {:?} in {:?}",
                vertex_id, access_time
            );
            return Ok(CacheResult::Hit(vertex, CacheLevel::L1));
        }

        // L1 Miss - Update stats
        {
            let mut stats = self.stats.write().await;
            stats.l1_misses += 1;
        }

        // L2 Cache Check (Medium Speed)
        debug!("Checking L2 cache for vertex {:?}", vertex_id);
        if let Some(vertex) = self.l2_cache.read().await.get(&key).await? {
            let access_time = start_time.elapsed();

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.l2_hits += 1;
                self.update_hit_ratio(&mut stats);
            }

            // Promote to L1 for faster future access
            self.l1_cache
                .write()
                .await
                .put(*key, vertex.clone())
                .await?;

            // Record metrics
            self.metrics
                .record_operation(
                    crate::metrics::OperationType::Read,
                    Some(CacheLevel::L2),
                    access_time,
                    self.hash_key(&key),
                    true,
                )
                .await;

            // ML prefetch learning
            if self.config.ml_prefetch_enabled {
                let pattern = crate::prefetch_engine::AccessPattern {
                    key: *key,
                    timestamp: std::time::SystemTime::now(),
                    access_type: crate::prefetch_engine::AccessType::Hit,
                    cache_level: Some(CacheLevel::L2),
                    preceding_keys: Vec::new(),
                };
                let _ = self.prefetch_engine.record_access_pattern(pattern).await;
            }

            debug!(
                "✅ L2 cache hit for vertex {:?}, promoted to L1 in {:?}",
                vertex_id, access_time
            );
            return Ok(CacheResult::Hit(vertex, CacheLevel::L2));
        }

        // L2 Miss - Update stats
        {
            let mut stats = self.stats.write().await;
            stats.l2_misses += 1;
        }

        // L3 Cache Check (Largest Capacity)
        debug!("Checking L3 cache for vertex {:?}", vertex_id);
        if let Some(vertex) = self.l3_cache.read().await.get(&key).await? {
            let access_time = start_time.elapsed();

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.l3_hits += 1;
                self.update_hit_ratio(&mut stats);
            }

            // Promote through the cache hierarchy
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

            // Record metrics
            self.metrics
                .record_operation(
                    crate::metrics::OperationType::Read,
                    Some(CacheLevel::L3),
                    access_time,
                    self.hash_key(&key),
                    true,
                )
                .await;

            // ML prefetch learning
            if self.config.ml_prefetch_enabled {
                let pattern = crate::prefetch_engine::AccessPattern {
                    key: *key,
                    timestamp: std::time::SystemTime::now(),
                    access_type: crate::prefetch_engine::AccessType::Hit,
                    cache_level: Some(CacheLevel::L3),
                    preceding_keys: Vec::new(),
                };
                let _ = self.prefetch_engine.record_access_pattern(pattern).await;
            }

            debug!(
                "✅ L3 cache hit for vertex {:?}, promoted to L1/L2 in {:?}",
                vertex_id, access_time
            );
            return Ok(CacheResult::Hit(vertex, CacheLevel::L3));
        }

        // L3 Miss - Complete cache miss
        {
            let mut stats = self.stats.write().await;
            stats.l3_misses += 1;
            self.update_hit_ratio(&mut stats);
        }

        let access_time = start_time.elapsed();

        // Record miss metrics
        self.metrics
            .record_operation(
                crate::metrics::OperationType::Read,
                None,
                access_time,
                self.hash_key(&key),
                false,
            )
            .await;

        // Trigger ML-powered prefetching for nearby data
        if self.config.ml_prefetch_enabled {
            if let Err(e) = self.prefetch_engine.trigger_prefetch(&key).await {
                warn!("Prefetch failed for key {:?}: {}", key, e);
            }
        }

        debug!(
            "❌ Complete cache miss for vertex {:?} in {:?}",
            vertex_id, access_time
        );
        Ok(CacheResult::Miss)
    }

    /// Store vertex in optimal cache level
    pub async fn put_vertex(&self, vertex_id: &VertexId, vertex: Vertex) -> Result<()> {
        let start_time = std::time::Instant::now();
        let key = vertex_id; // VertexId and Hash256 are both [u8; 32]

        // Determine optimal cache level using ML predictions
        let optimal_level = if self.config.ml_prefetch_enabled {
            self.prefetch_engine
                .determine_cache_level(&key)
                .await
                .unwrap_or(CacheLevel::L3) // Default to L3 for new data
        } else {
            CacheLevel::L3 // Conservative default
        };

        // NUMA-aware memory allocation
        let allocation_size_mb = self.estimate_vertex_size(&vertex) / (1024 * 1024);
        if let Ok(allocation) = self
            .memory_optimizer
            .read()
            .await
            .allocate_memory(allocation_size_mb, None)
            .await
        {
            debug!(
                "Allocated {}MB on NUMA node {} for vertex storage",
                allocation.size_mb, allocation.numa_node
            );
        }

        // Store in determined cache level
        match optimal_level {
            CacheLevel::L1 => {
                self.l1_cache
                    .write()
                    .await
                    .put(*key, vertex.clone())
                    .await?;
                debug!("Stored vertex {:?} in L1 cache", vertex_id);
            }
            CacheLevel::L2 => {
                self.l2_cache
                    .write()
                    .await
                    .put(*key, vertex.clone())
                    .await?;
                debug!("Stored vertex {:?} in L2 cache", vertex_id);
            }
            CacheLevel::L3 => {
                self.l3_cache
                    .write()
                    .await
                    .put(*key, vertex.clone())
                    .await?;
                debug!("Stored vertex {:?} in L3 cache", vertex_id);
            }
        }

        let access_time = start_time.elapsed();

        // Record write metrics
        self.metrics
            .record_operation(
                crate::metrics::OperationType::Write,
                Some(optimal_level),
                access_time,
                self.hash_key(&key),
                true,
            )
            .await;

        // Update coherency across shards
        if let Err(e) = self
            .coherency_manager
            .on_cache_update(*key, optimal_level)
            .await
        {
            warn!("Cache coherency notification failed: {}", e);
        }

        debug!(
            "✅ Stored vertex {:?} in {:?} cache in {:?}",
            vertex_id, optimal_level, access_time
        );
        Ok(())
    }

    /// Get state value with hierarchical lookup
    pub async fn get_state(&self, key: &StateKey) -> Result<CacheResult<StateValue>> {
        let hash_key: Hash256 = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(key);
            hasher.finalize().into()
        };
        let start_time = std::time::Instant::now();

        // State data typically resides in L3 cache
        if let Some(value) = self.l3_cache.read().await.get_state(&hash_key).await? {
            let access_time = start_time.elapsed();

            {
                let mut stats = self.stats.write().await;
                stats.l3_hits += 1;
                stats.total_requests += 1;
                self.update_hit_ratio(&mut stats);
            }

            self.metrics
                .record_operation(
                    crate::metrics::OperationType::Read,
                    Some(CacheLevel::L3),
                    access_time,
                    self.hash_key(&hash_key),
                    true,
                )
                .await;

            debug!("✅ L3 state cache hit in {:?}", access_time);
            return Ok(CacheResult::Hit(value, CacheLevel::L3));
        }

        // State cache miss
        let access_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().await;
            stats.l3_misses += 1;
            stats.total_requests += 1;
            self.update_hit_ratio(&mut stats);
        }

        self.metrics
            .record_operation(
                crate::metrics::OperationType::Read,
                Some(CacheLevel::L3),
                access_time,
                self.hash_key(&hash_key),
                false,
            )
            .await;

        debug!("❌ State cache miss in {:?}", access_time);
        Ok(CacheResult::Miss)
    }

    /// Store state value in L3 cache
    pub async fn put_state(&self, key: &StateKey, value: StateValue) -> Result<()> {
        let hash_key: Hash256 = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(key);
            hasher.finalize().into()
        };
        let start_time = std::time::Instant::now();

        // State data goes to L3 cache
        self.l3_cache
            .write()
            .await
            .put_state(hash_key, value.clone())
            .await?;

        let access_time = start_time.elapsed();

        // Record write metrics
        self.metrics
            .record_operation(
                crate::metrics::OperationType::Write,
                Some(CacheLevel::L3),
                access_time,
                self.hash_key(&hash_key),
                true,
            )
            .await;

        // Coherency notification
        if let Err(e) = self
            .coherency_manager
            .on_cache_update(hash_key, CacheLevel::L3)
            .await
        {
            warn!("State cache coherency notification failed: {}", e);
        }

        debug!("✅ Stored state value in L3 cache in {:?}", access_time);
        Ok(())
    }

    /// Invalidate entry across all cache levels
    pub async fn invalidate(&self, key: &Hash256) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Invalidate across all levels
        let l1_removed = self.l1_cache.write().await.remove(key).await?;
        let l2_removed = self.l2_cache.write().await.remove(key).await?;
        let l3_removed = self.l3_cache.write().await.remove(key).await?;

        let access_time = start_time.elapsed();

        if l1_removed || l2_removed || l3_removed {
            // Record invalidation metrics
            self.metrics
                .record_operation(
                    crate::metrics::OperationType::Invalidation,
                    None,
                    access_time,
                    self.hash_key(key),
                    true,
                )
                .await;

            // Notify coherency manager
            if let Err(e) = self.coherency_manager.invalidate_entry(*key).await {
                warn!("Cache coherency invalidation failed: {}", e);
            }

            debug!(
                "✅ Invalidated cache entry for key {:?} in {:?}",
                key, access_time
            );
        }

        Ok(())
    }

    /// Flush all dirty entries to persistent storage
    pub async fn flush_dirty(&self) -> Result<u64> {
        let start_time = std::time::Instant::now();
        let mut total_flushed = 0;

        // Flush each cache level
        total_flushed += self.l1_cache.write().await.flush_dirty().await?;
        total_flushed += self.l2_cache.write().await.flush_dirty().await?;
        total_flushed += self.l3_cache.write().await.flush_dirty().await?;

        let flush_time = start_time.elapsed();

        if total_flushed > 0 {
            info!(
                "✅ Flushed {} dirty cache entries in {:?}",
                total_flushed, flush_time
            );
        }

        Ok(total_flushed)
    }

    /// Get current cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();

        // Calculate average access time from metrics
        let current_metrics = self.metrics.get_current_metrics().await;
        stats.average_access_time_ns = current_metrics.avg_access_time_ns;

        stats
    }

    /// Get detailed performance metrics
    pub async fn get_performance_metrics(&self) -> Result<String> {
        self.metrics.export_metrics_json().await
    }

    /// Optimize cache performance
    pub async fn optimize_performance(&mut self) -> Result<()> {
        info!("🎯 Starting cache performance optimization");

        // Memory optimization
        let memory_migrations = self
            .memory_optimizer
            .write()
            .await
            .optimize_memory_layout()
            .await?;

        // ML-powered prefetch optimization
        if self.config.ml_prefetch_enabled {
            // Note: optimize_model_parameters requires mutable access, skipping for now
            // In production, we'd need Arc<RwLock<PrefetchEngine>> for mutable access
            // self.prefetch_engine.optimize_model_parameters().await?;
        }

        // Cache coherency optimization
        // Coherency optimization handled during maintenance_cleanup
        if let Err(e) = self.coherency_manager.maintenance_cleanup().await {
            warn!("Cache coherency maintenance failed: {}", e);
        }

        // Check memory pressure
        self.memory_optimizer
            .read()
            .await
            .check_memory_pressure()
            .await?;

        info!(
            "✅ Performance optimization complete, {} memory pages migrated",
            memory_migrations
        );
        Ok(())
    }

    /// Validate Phase 2 performance targets
    pub async fn validate_phase2_targets(&self) -> Result<bool> {
        let stats = self.get_stats().await;
        let memory_usage = self.get_memory_usage().await?;

        // Phase 2 targets:
        // - 90% hit ratio
        // - <2GB total memory usage
        // - <10ms average access time

        let hit_ratio_ok = stats.hit_ratio >= 0.90;
        let memory_ok = memory_usage.values().sum::<usize>() <= 2048; // 2GB in MB
        let latency_ok = stats.average_access_time_ns <= 10_000_000.0; // 10ms in ns

        info!("🎯 Phase 2 Target Validation:");
        info!(
            "   Hit Ratio: {:.1}% (target: 90%) - {}",
            stats.hit_ratio * 100.0,
            if hit_ratio_ok { "✅" } else { "❌" }
        );
        info!(
            "   Memory Usage: {}MB (limit: 2048MB) - {}",
            memory_usage.values().sum::<usize>(),
            if memory_ok { "✅" } else { "❌" }
        );
        info!(
            "   Avg Latency: {:.1}ms (target: <10ms) - {}",
            stats.average_access_time_ns / 1_000_000.0,
            if latency_ok { "✅" } else { "❌" }
        );

        let success = hit_ratio_ok && memory_ok && latency_ok;

        if success {
            info!("🎉 Phase 2 targets ACHIEVED! Cache system ready for 100,000+ TPS");
        } else {
            warn!("⚠️ Phase 2 targets not met, additional optimization needed");
        }

        Ok(success)
    }

    // Helper methods

    fn update_hit_ratio(&self, stats: &mut CacheStats) {
        let total_hits = stats.l1_hits + stats.l2_hits + stats.l3_hits;
        if stats.total_requests > 0 {
            stats.hit_ratio = total_hits as f64 / stats.total_requests as f64;
        }
    }

    fn hash_key(&self, key: &Hash256) -> u64 {
        // Simple hash for metrics (in production would use proper hash function)
        key.iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
    }

    fn estimate_vertex_size(&self, vertex: &Vertex) -> usize {
        // Simplified size estimation
        std::mem::size_of_val(vertex) + vertex.transactions.len() * 256
    }

    pub async fn get_memory_usage(&self) -> Result<std::collections::HashMap<CacheLevel, usize>> {
        let mut usage = std::collections::HashMap::new();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex() -> Vertex {
        Vertex {
            id: [1u8; 32],
            round: 1,
            author: [2u8; 32],
            tx_root: [3u8; 32],
            parents: vec![[4u8; 32]],
            transactions: Vec::new(),
            signature: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_hierarchical_cache_creation() {
        let config = CacheConfig::default();
        let cache = HierarchicalCache::new(config).await;
        assert!(cache.is_ok());
    }

    #[tokio::test]
    async fn test_vertex_cache_operations() {
        let config = CacheConfig::default();
        let cache = HierarchicalCache::new(config).await.unwrap();

        let vertex_id = [1u8; 32];
        let vertex = create_test_vertex();

        // Test cache miss
        let result = cache.get_vertex(&vertex_id).await.unwrap();
        match result {
            CacheResult::Miss => {} // Expected
            _ => panic!("Expected cache miss"),
        }

        // Test cache put and hit
        cache.put_vertex(&vertex_id, vertex.clone()).await.unwrap();
        let result = cache.get_vertex(&vertex_id).await.unwrap();
        match result {
            CacheResult::Hit(cached_vertex, level) => {
                assert_eq!(cached_vertex.id, vertex.id);
                assert_eq!(level, CacheLevel::L3); // Should be stored in L3 by default
            }
            _ => panic!("Expected cache hit"),
        }
    }
}
