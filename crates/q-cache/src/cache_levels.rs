// Q-NarwhalKnight Cache Levels Implementation
// L1/L2/L3 cache hierarchy for optimized data access

use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use hex;
use parking_lot::Mutex;
use q_types::{Hash256, NodeId, Round, StateKey, StateValue, Vertex, VertexId};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// L1 Cache - Hot Vertex Cache (1MB, fastest access)
#[derive(Debug)]
pub struct L1Cache {
    max_size_bytes: usize,
    current_size_bytes: usize,
    entries: DashMap<Hash256, L1CacheEntry>,
    access_order: Arc<Mutex<Vec<Hash256>>>, // LRU tracking
    stats: L1Stats,
}

/// L1 cache entry with metadata
#[derive(Debug, Clone)]
struct L1CacheEntry {
    vertex: Vertex,
    access_count: u32,
    last_accessed: SystemTime,
    size_bytes: usize,
}

/// L1 cache statistics
#[derive(Debug, Clone, Default)]
struct L1Stats {
    hits: u64,
    misses: u64,
    evictions: u64,
    average_access_time_ns: f64,
}

impl L1Cache {
    /// Create new L1 cache with specified size
    pub async fn new(size_mb: usize) -> Result<Self> {
        let max_size_bytes = size_mb * 1024 * 1024;

        tracing::info!("Creating L1 cache with {}MB capacity", size_mb);

        Ok(Self {
            max_size_bytes,
            current_size_bytes: 0,
            entries: DashMap::new(),
            access_order: Arc::new(Mutex::new(Vec::new())),
            stats: L1Stats::default(),
        })
    }

    /// Get vertex from L1 cache
    pub async fn get(&self, key: &Hash256) -> Result<Option<Vertex>> {
        let start_time = std::time::Instant::now();

        if let Some(mut entry_ref) = self.entries.get_mut(key) {
            // Update access metadata
            entry_ref.access_count += 1;
            entry_ref.last_accessed = SystemTime::now();

            // Update LRU order
            {
                let mut order = self.access_order.lock();
                if let Some(pos) = order.iter().position(|k| k == key) {
                    order.remove(pos);
                }
                order.push(*key);
            }

            let vertex = entry_ref.vertex.clone();
            let access_time = start_time.elapsed();

            // Update stats (simplified - would use atomic operations in production)
            tracing::debug!(
                "L1 cache hit for key {:?} in {:?}",
                hex::encode(key),
                access_time
            );

            return Ok(Some(vertex));
        }

        tracing::debug!("L1 cache miss for key {:?}", hex::encode(key));
        Ok(None)
    }

    /// Store vertex in L1 cache
    pub async fn put(&mut self, key: Hash256, vertex: Vertex) -> Result<()> {
        let vertex_size = self.estimate_vertex_size(&vertex);

        // Evict entries if necessary
        while self.current_size_bytes + vertex_size > self.max_size_bytes
            && !self.entries.is_empty()
        {
            self.evict_lru_entry().await?;
        }

        // Only insert if it fits
        if vertex_size <= self.max_size_bytes {
            let entry = L1CacheEntry {
                vertex,
                access_count: 1,
                last_accessed: SystemTime::now(),
                size_bytes: vertex_size,
            };

            self.entries.insert(key, entry);
            self.current_size_bytes += vertex_size;

            // Update LRU order
            {
                let mut order = self.access_order.lock();
                order.push(key);
            }

            tracing::debug!(
                "Stored vertex in L1 cache, size: {}MB",
                self.current_size_bytes / (1024 * 1024)
            );
        }

        Ok(())
    }

    /// Remove entry from L1 cache
    pub async fn remove(&mut self, key: &Hash256) -> Result<bool> {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.current_size_bytes -= entry.size_bytes;

            // Remove from LRU order
            {
                let mut order = self.access_order.lock();
                if let Some(pos) = order.iter().position(|k| k == key) {
                    order.remove(pos);
                }
            }

            tracing::debug!("Removed entry from L1 cache");
            return Ok(true);
        }
        Ok(false)
    }

    /// Evict least recently used entry
    async fn evict_lru_entry(&mut self) -> Result<()> {
        let key_to_evict = {
            let order = self.access_order.lock();
            order.first().copied()
        };

        if let Some(key) = key_to_evict {
            self.remove(&key).await?;
            tracing::debug!("Evicted LRU entry from L1 cache: {:?}", hex::encode(&key));
        }

        Ok(())
    }

    /// Estimate vertex size in bytes (simplified)
    fn estimate_vertex_size(&self, vertex: &Vertex) -> usize {
        // Simplified size estimation
        vertex.transactions.len() * 256 + 512 // Rough estimate
    }

    /// Flush dirty entries (L1 cache is typically read-only)
    pub async fn flush_dirty(&mut self) -> Result<u64> {
        // L1 cache typically doesn't have dirty entries
        Ok(0)
    }

    /// Get memory usage in bytes
    pub async fn memory_usage(&self) -> Result<usize> {
        Ok(self.current_size_bytes)
    }
}

/// L2 Cache - Block Cache (100MB, medium speed)
#[derive(Debug)]
pub struct L2Cache {
    max_size_bytes: usize,
    current_size_bytes: usize,
    entries: Arc<RwLock<HashMap<Hash256, L2CacheEntry>>>,
    access_frequency: Arc<RwLock<BTreeMap<u32, Vec<Hash256>>>>, // Frequency-based eviction
    stats: L2Stats,
}

/// L2 cache entry with block-level metadata
#[derive(Debug, Clone)]
struct L2CacheEntry {
    vertex: Vertex,
    access_count: u32,
    last_accessed: SystemTime,
    block_height: u64,
    size_bytes: usize,
    is_dirty: bool,
}

/// L2 cache statistics
#[derive(Debug, Clone, Default)]
struct L2Stats {
    hits: u64,
    misses: u64,
    evictions: u64,
    dirty_evictions: u64,
}

impl L2Cache {
    /// Create new L2 cache
    pub async fn new(size_mb: usize) -> Result<Self> {
        let max_size_bytes = size_mb * 1024 * 1024;

        tracing::info!("Creating L2 cache with {}MB capacity", size_mb);

        Ok(Self {
            max_size_bytes,
            current_size_bytes: 0,
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_frequency: Arc::new(RwLock::new(BTreeMap::new())),
            stats: L2Stats::default(),
        })
    }

    /// Get vertex from L2 cache
    pub async fn get(&self, key: &Hash256) -> Result<Option<Vertex>> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get_mut(key) {
            // Update access metadata
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();

            let vertex = entry.vertex.clone();

            tracing::debug!("L2 cache hit for key {:?}", hex::encode(key));
            return Ok(Some(vertex));
        }

        tracing::debug!("L2 cache miss for key {:?}", hex::encode(key));
        Ok(None)
    }

    /// Store vertex in L2 cache
    pub async fn put(&mut self, key: Hash256, vertex: Vertex) -> Result<()> {
        let vertex_size = self.estimate_vertex_size(&vertex);

        // Evict entries if necessary
        while self.current_size_bytes + vertex_size > self.max_size_bytes {
            self.evict_least_frequent_entry().await?;
        }

        let entry = L2CacheEntry {
            vertex,
            access_count: 1,
            last_accessed: SystemTime::now(),
            block_height: 0, // Would be set from actual block data
            size_bytes: vertex_size,
            is_dirty: false,
        };

        let mut entries = self.entries.write().await;
        entries.insert(key, entry);
        self.current_size_bytes += vertex_size;

        tracing::debug!("Stored vertex in L2 cache");
        Ok(())
    }

    /// Remove entry from L2 cache
    pub async fn remove(&mut self, key: &Hash256) -> Result<bool> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.remove(key) {
            self.current_size_bytes -= entry.size_bytes;
            tracing::debug!("Removed entry from L2 cache");
            return Ok(true);
        }
        Ok(false)
    }

    /// Evict least frequently used entry
    async fn evict_least_frequent_entry(&mut self) -> Result<()> {
        let mut entries = self.entries.write().await;

        // Find entry with lowest access count
        let key_to_evict = entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(k, _)| *k);

        if let Some(key) = key_to_evict {
            if let Some(entry) = entries.remove(&key) {
                self.current_size_bytes -= entry.size_bytes;
                tracing::debug!(
                    "Evicted least frequent entry from L2 cache: {:?}",
                    hex::encode(&key)
                );
            }
        }

        Ok(())
    }

    /// Estimate vertex size
    fn estimate_vertex_size(&self, vertex: &Vertex) -> usize {
        vertex.transactions.len() * 256 + 512
    }

    /// Flush dirty entries
    pub async fn flush_dirty(&mut self) -> Result<u64> {
        let entries = self.entries.read().await;
        let dirty_count = entries.values().filter(|e| e.is_dirty).count() as u64;

        // In real implementation, would write dirty entries to storage
        tracing::debug!("L2 cache has {} dirty entries to flush", dirty_count);

        Ok(dirty_count)
    }

    /// Get memory usage
    pub async fn memory_usage(&self) -> Result<usize> {
        Ok(self.current_size_bytes)
    }
}

/// L3 Cache - State Cache (1GB, largest capacity)
#[derive(Debug)]
pub struct L3Cache {
    max_size_bytes: usize,
    current_size_bytes: usize,
    vertex_entries: Arc<RwLock<HashMap<Hash256, L3VertexEntry>>>,
    state_entries: Arc<RwLock<HashMap<Hash256, L3StateEntry>>>,
    eviction_queue: Arc<RwLock<Vec<(Hash256, SystemTime)>>>,
    stats: L3Stats,
}

/// L3 vertex cache entry
#[derive(Debug, Clone)]
struct L3VertexEntry {
    vertex: Vertex,
    access_count: u32,
    last_accessed: SystemTime,
    size_bytes: usize,
    is_dirty: bool,
}

/// L3 state cache entry
#[derive(Debug, Clone)]
struct L3StateEntry {
    value: StateValue,
    access_count: u32,
    last_accessed: SystemTime,
    size_bytes: usize,
    is_dirty: bool,
}

/// L3 cache statistics
#[derive(Debug, Clone, Default)]
struct L3Stats {
    vertex_hits: u64,
    vertex_misses: u64,
    state_hits: u64,
    state_misses: u64,
    evictions: u64,
}

impl L3Cache {
    /// Create new L3 cache
    pub async fn new(size_mb: usize) -> Result<Self> {
        let max_size_bytes = size_mb * 1024 * 1024;

        tracing::info!("Creating L3 cache with {}MB capacity", size_mb);

        Ok(Self {
            max_size_bytes,
            current_size_bytes: 0,
            vertex_entries: Arc::new(RwLock::new(HashMap::new())),
            state_entries: Arc::new(RwLock::new(HashMap::new())),
            eviction_queue: Arc::new(RwLock::new(Vec::new())),
            stats: L3Stats::default(),
        })
    }

    /// Get vertex from L3 cache
    pub async fn get(&self, key: &Hash256) -> Result<Option<Vertex>> {
        let mut entries = self.vertex_entries.write().await;

        if let Some(entry) = entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();

            let vertex = entry.vertex.clone();
            tracing::debug!("L3 vertex cache hit for key {:?}", hex::encode(key));
            return Ok(Some(vertex));
        }

        Ok(None)
    }

    /// Get state from L3 cache  
    pub async fn get_state(&self, key: &Hash256) -> Result<Option<StateValue>> {
        let mut entries = self.state_entries.write().await;

        if let Some(entry) = entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();

            let value = entry.value.clone();
            tracing::debug!("L3 state cache hit for key {:?}", hex::encode(key));
            return Ok(Some(value));
        }

        Ok(None)
    }

    /// Store vertex in L3 cache
    pub async fn put(&mut self, key: Hash256, vertex: Vertex) -> Result<()> {
        let vertex_size = self.estimate_vertex_size(&vertex);

        // Evict if necessary
        self.ensure_capacity(vertex_size).await?;

        let entry = L3VertexEntry {
            vertex,
            access_count: 1,
            last_accessed: SystemTime::now(),
            size_bytes: vertex_size,
            is_dirty: false,
        };

        let mut entries = self.vertex_entries.write().await;
        entries.insert(key, entry);
        self.current_size_bytes += vertex_size;

        tracing::debug!("Stored vertex in L3 cache");
        Ok(())
    }

    /// Store state in L3 cache
    pub async fn put_state(&mut self, key: Hash256, value: StateValue) -> Result<()> {
        let value_size = value.len();

        // Evict if necessary
        self.ensure_capacity(value_size).await?;

        let entry = L3StateEntry {
            value,
            access_count: 1,
            last_accessed: SystemTime::now(),
            size_bytes: value_size,
            is_dirty: false,
        };

        let mut entries = self.state_entries.write().await;
        entries.insert(key, entry);
        self.current_size_bytes += value_size;

        tracing::debug!("Stored state in L3 cache");
        Ok(())
    }

    /// Remove entry from L3 cache
    pub async fn remove(&mut self, key: &Hash256) -> Result<bool> {
        let mut removed = false;

        // Try removing from vertex entries
        {
            let mut entries = self.vertex_entries.write().await;
            if let Some(entry) = entries.remove(key) {
                self.current_size_bytes -= entry.size_bytes;
                removed = true;
            }
        }

        // Try removing from state entries
        if !removed {
            let mut entries = self.state_entries.write().await;
            if let Some(entry) = entries.remove(key) {
                self.current_size_bytes -= entry.size_bytes;
                removed = true;
            }
        }

        if removed {
            tracing::debug!("Removed entry from L3 cache");
        }

        Ok(removed)
    }

    /// Ensure cache has capacity for new entry
    async fn ensure_capacity(&mut self, required_bytes: usize) -> Result<()> {
        while self.current_size_bytes + required_bytes > self.max_size_bytes {
            self.evict_oldest_entry().await?;
        }
        Ok(())
    }

    /// Evict oldest entry from L3 cache
    async fn evict_oldest_entry(&mut self) -> Result<()> {
        let oldest_key = {
            let vertex_entries = self.vertex_entries.read().await;
            let state_entries = self.state_entries.read().await;

            // Find oldest entry across both maps
            let vertex_oldest = vertex_entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(k, entry)| (*k, entry.last_accessed));

            let state_oldest = state_entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(k, entry)| (*k, entry.last_accessed));

            match (vertex_oldest, state_oldest) {
                (Some((vk, vt)), Some((sk, st))) => {
                    if vt < st {
                        Some(vk)
                    } else {
                        Some(sk)
                    }
                }
                (Some((vk, _)), None) => Some(vk),
                (None, Some((sk, _))) => Some(sk),
                (None, None) => None,
            }
        };

        if let Some(key) = oldest_key {
            self.remove(&key).await?;
            tracing::debug!(
                "Evicted oldest entry from L3 cache: {:?}",
                hex::encode(&key)
            );
        }

        Ok(())
    }

    /// Estimate vertex size
    fn estimate_vertex_size(&self, vertex: &Vertex) -> usize {
        vertex.transactions.len() * 256 + 512
    }

    /// Flush dirty entries
    pub async fn flush_dirty(&mut self) -> Result<u64> {
        let vertex_entries = self.vertex_entries.read().await;
        let state_entries = self.state_entries.read().await;

        let vertex_dirty = vertex_entries.values().filter(|e| e.is_dirty).count();
        let state_dirty = state_entries.values().filter(|e| e.is_dirty).count();
        let total_dirty = vertex_dirty + state_dirty;

        tracing::debug!("L3 cache has {} dirty entries to flush", total_dirty);

        Ok(total_dirty as u64)
    }

    /// Get memory usage
    pub async fn memory_usage(&self) -> Result<usize> {
        Ok(self.current_size_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn test_l1_cache_operations() {
        let mut cache = L1Cache::new(1).await.unwrap(); // 1MB

        let key = [5u8; 32];
        let vertex = create_test_vertex();

        // Test miss
        assert!(cache.get(&key).await.unwrap().is_none());

        // Test put and hit
        cache.put(key, vertex.clone()).await.unwrap();
        assert!(cache.get(&key).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_l2_cache_operations() {
        let mut cache = L2Cache::new(10).await.unwrap(); // 10MB

        let key = [6u8; 32];
        let vertex = create_test_vertex();

        // Test miss
        assert!(cache.get(&key).await.unwrap().is_none());

        // Test put and hit
        cache.put(key, vertex.clone()).await.unwrap();
        assert!(cache.get(&key).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_l3_cache_operations() {
        let mut cache = L3Cache::new(100).await.unwrap(); // 100MB

        let key = [7u8; 32];
        let vertex = create_test_vertex();
        let state_value = vec![1, 2, 3, 4];

        // Test vertex operations
        assert!(cache.get(&key).await.unwrap().is_none());
        cache.put(key, vertex).await.unwrap();
        assert!(cache.get(&key).await.unwrap().is_some());

        // Test state operations
        let state_key = [8u8; 32];
        assert!(cache.get_state(&state_key).await.unwrap().is_none());
        cache
            .put_state(state_key, state_value.clone())
            .await
            .unwrap();
        assert_eq!(
            cache.get_state(&state_key).await.unwrap().unwrap(),
            state_value
        );
    }
}
