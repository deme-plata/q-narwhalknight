//! Sharded Balance Cache for High-Performance Balance Lookups
//!
//! v3.4.6-beta: Ported from QTFT blockchain for 2-3x balance lookup speedup.
//!
//! ## Architecture
//!
//! Uses 16 DashMap shards to distribute lock contention across CPU cores.
//! Each shard handles ~6.25% of addresses based on hash distribution.
//!
//! ## Performance
//!
//! - **Before**: Single RwLock causes contention at high throughput
//! - **After**: 16 parallel shards = 16x reduction in lock contention
//! - **Expected Speedup**: 2-3x for balance lookups under load
//!
//! ## Usage
//!
//! ```rust
//! let cache = ShardedBalanceCache::new();
//! cache.insert([0u8; 32], 1_000_000);
//! let balance = cache.get(&[0u8; 32]);
//! ```

use dashmap::DashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use tracing::{debug, trace};

/// Number of shards for lock distribution
/// 16 is optimal for most CPUs (2-16 cores)
const NUM_SHARDS: usize = 16;

/// High-performance sharded balance cache
///
/// Distributes balance lookups across 16 DashMap shards to minimize
/// lock contention under high throughput (10,000+ TPS).
#[derive(Debug)]
pub struct ShardedBalanceCache {
    /// 16 shards indexed by address hash
    shards: Vec<DashMap<[u8; 32], u128>>,

    /// Total number of entries across all shards
    total_entries: AtomicUsize,

    /// Cache hits for metrics
    cache_hits: AtomicU64,

    /// Cache misses for metrics
    cache_misses: AtomicU64,
}

impl ShardedBalanceCache {
    /// Create new sharded balance cache
    pub fn new() -> Self {
        Self {
            shards: (0..NUM_SHARDS)
                .map(|_| DashMap::with_capacity(10_000))
                .collect(),
            total_entries: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Create with pre-allocated capacity per shard
    pub fn with_capacity(capacity_per_shard: usize) -> Self {
        Self {
            shards: (0..NUM_SHARDS)
                .map(|_| DashMap::with_capacity(capacity_per_shard))
                .collect(),
            total_entries: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Calculate shard index from address (deterministic hash)
    #[inline]
    fn shard_index(address: &[u8; 32]) -> usize {
        let mut hasher = DefaultHasher::new();
        address.hash(&mut hasher);
        (hasher.finish() as usize) % NUM_SHARDS
    }

    /// Get balance for address (O(1) average)
    #[inline]
    pub fn get(&self, address: &[u8; 32]) -> Option<u128> {
        let shard = &self.shards[Self::shard_index(address)];
        let result = shard.get(address).map(|v| *v);

        if result.is_some() {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            trace!("💰 [SHARDED CACHE] Hit for address {:?}", &address[..4]);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Insert or update balance for address
    #[inline]
    pub fn insert(&self, address: [u8; 32], balance: u128) {
        let shard = &self.shards[Self::shard_index(&address)];
        let was_new = shard.insert(address, balance).is_none();

        if was_new {
            self.total_entries.fetch_add(1, Ordering::Relaxed);
        }

        trace!("💰 [SHARDED CACHE] {} balance {} for {:?}",
               if was_new { "Inserted" } else { "Updated" },
               balance,
               &address[..4]);
    }

    /// Update balance using a function (atomic read-modify-write)
    #[inline]
    pub fn update<F>(&self, address: &[u8; 32], f: F) -> Option<u128>
    where
        F: FnOnce(u128) -> u128,
    {
        let shard = &self.shards[Self::shard_index(address)];

        shard.get_mut(address).map(|mut entry| {
            let new_value = f(*entry);
            *entry = new_value;
            new_value
        })
    }

    /// Add amount to balance (creates if not exists)
    #[inline]
    pub fn add_balance(&self, address: &[u8; 32], amount: u128) -> u128 {
        let shard = &self.shards[Self::shard_index(address)];

        let mut entry = shard.entry(*address).or_insert(0);
        *entry = entry.saturating_add(amount);
        let new_balance = *entry;

        // Update entry count if this was a new entry
        if new_balance == amount {
            self.total_entries.fetch_add(1, Ordering::Relaxed);
        }

        debug!("💰 [SHARDED CACHE] Added {} to {:?}, new balance: {}",
               amount, &address[..4], new_balance);

        new_balance
    }

    /// Subtract amount from balance (returns None if insufficient)
    #[inline]
    pub fn subtract_balance(&self, address: &[u8; 32], amount: u128) -> Option<u128> {
        let shard = &self.shards[Self::shard_index(address)];

        let mut entry = shard.get_mut(address)?;

        if *entry < amount {
            return None; // Insufficient balance
        }

        *entry = entry.saturating_sub(amount);
        let new_balance = *entry;

        debug!("💰 [SHARDED CACHE] Subtracted {} from {:?}, new balance: {}",
               amount, &address[..4], new_balance);

        Some(new_balance)
    }

    /// Remove address from cache
    #[inline]
    pub fn remove(&self, address: &[u8; 32]) -> Option<u128> {
        let shard = &self.shards[Self::shard_index(address)];
        let result = shard.remove(address).map(|(_, v)| v);

        if result.is_some() {
            self.total_entries.fetch_sub(1, Ordering::Relaxed);
        }

        result
    }

    /// Check if address exists in cache
    #[inline]
    pub fn contains(&self, address: &[u8; 32]) -> bool {
        let shard = &self.shards[Self::shard_index(address)];
        shard.contains_key(address)
    }

    /// Get total number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.total_entries.load(Ordering::Relaxed)
    }

    /// Check if cache is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
        self.total_entries.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> ShardedCacheStats {
        let total = self.len();
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total_ops = hits + misses;

        let hit_rate = if total_ops > 0 {
            (hits as f64 / total_ops as f64) * 100.0
        } else {
            0.0
        };

        // Calculate per-shard distribution
        let shard_sizes: Vec<usize> = self.shards.iter().map(|s| s.len()).collect();
        let min_shard = *shard_sizes.iter().min().unwrap_or(&0);
        let max_shard = *shard_sizes.iter().max().unwrap_or(&0);

        ShardedCacheStats {
            total_entries: total,
            cache_hits: hits,
            cache_misses: misses,
            hit_rate_percent: hit_rate,
            shard_count: NUM_SHARDS,
            min_shard_size: min_shard,
            max_shard_size: max_shard,
        }
    }

    /// Iterate over all entries (for backup/persistence)
    /// Note: This acquires locks on all shards sequentially
    pub fn iter_all(&self) -> impl Iterator<Item = ([u8; 32], u128)> + '_ {
        self.shards.iter().flat_map(|shard| {
            shard.iter().map(|entry| (*entry.key(), *entry.value()))
        })
    }

    /// Get balances for multiple addresses (batch operation)
    pub fn get_batch(&self, addresses: &[[u8; 32]]) -> Vec<([u8; 32], Option<u128>)> {
        addresses
            .iter()
            .map(|addr| (*addr, self.get(addr)))
            .collect()
    }

    /// Insert multiple balances (batch operation)
    pub fn insert_batch(&self, entries: &[([u8; 32], u128)]) {
        for (address, balance) in entries {
            self.insert(*address, *balance);
        }
    }
}

impl Default for ShardedBalanceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct ShardedCacheStats {
    /// Total entries across all shards
    pub total_entries: usize,
    /// Total cache hits
    pub cache_hits: u64,
    /// Total cache misses
    pub cache_misses: u64,
    /// Hit rate as percentage
    pub hit_rate_percent: f64,
    /// Number of shards
    pub shard_count: usize,
    /// Size of smallest shard
    pub min_shard_size: usize,
    /// Size of largest shard
    pub max_shard_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let cache = ShardedBalanceCache::new();

        let addr1 = [1u8; 32];
        let addr2 = [2u8; 32];

        // Test insert and get
        cache.insert(addr1, 1000);
        assert_eq!(cache.get(&addr1), Some(1000));
        assert_eq!(cache.get(&addr2), None);

        // Test update
        cache.insert(addr1, 2000);
        assert_eq!(cache.get(&addr1), Some(2000));

        // Test len
        assert_eq!(cache.len(), 1);
        cache.insert(addr2, 500);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_add_subtract_balance() {
        let cache = ShardedBalanceCache::new();
        let addr = [42u8; 32];

        // Add to non-existent creates
        assert_eq!(cache.add_balance(&addr, 100), 100);
        assert_eq!(cache.get(&addr), Some(100));

        // Add to existing
        assert_eq!(cache.add_balance(&addr, 50), 150);

        // Subtract valid
        assert_eq!(cache.subtract_balance(&addr, 30), Some(120));

        // Subtract insufficient fails
        assert_eq!(cache.subtract_balance(&addr, 200), None);
        assert_eq!(cache.get(&addr), Some(120)); // Unchanged
    }

    #[test]
    fn test_update_function() {
        let cache = ShardedBalanceCache::new();
        let addr = [7u8; 32];

        cache.insert(addr, 1000);

        // Double the balance
        let new_val = cache.update(&addr, |v| v * 2);
        assert_eq!(new_val, Some(2000));
        assert_eq!(cache.get(&addr), Some(2000));

        // Update non-existent returns None
        let missing = [99u8; 32];
        assert_eq!(cache.update(&missing, |v| v + 1), None);
    }

    #[test]
    fn test_shard_distribution() {
        let cache = ShardedBalanceCache::new();

        // Insert many addresses
        for i in 0u8..100 {
            let mut addr = [0u8; 32];
            addr[0] = i;
            cache.insert(addr, i as u128 * 100);
        }

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 100);
        assert_eq!(stats.shard_count, 16);

        // Check distribution is reasonably even
        // Each shard should have ~6-7 entries on average
        assert!(stats.max_shard_size < 20, "Shard distribution too uneven");
    }

    #[test]
    fn test_batch_operations() {
        let cache = ShardedBalanceCache::new();

        let entries: Vec<([u8; 32], u128)> = (0u8..10)
            .map(|i| {
                let mut addr = [0u8; 32];
                addr[0] = i;
                (addr, i as u128 * 1000)
            })
            .collect();

        cache.insert_batch(&entries);
        assert_eq!(cache.len(), 10);

        let addresses: Vec<[u8; 32]> = entries.iter().map(|(a, _)| *a).collect();
        let results = cache.get_batch(&addresses);

        assert_eq!(results.len(), 10);
        for (addr, balance) in results {
            assert!(balance.is_some());
        }
    }

    #[test]
    fn test_clear() {
        let cache = ShardedBalanceCache::new();

        for i in 0u8..50 {
            let mut addr = [0u8; 32];
            addr[0] = i;
            cache.insert(addr, 1000);
        }

        assert_eq!(cache.len(), 50);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(ShardedBalanceCache::new());
        let mut handles = vec![];

        // Spawn 10 threads each inserting 100 entries
        for t in 0u8..10 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0u8..100 {
                    let mut addr = [0u8; 32];
                    addr[0] = t;
                    addr[1] = i;
                    cache_clone.insert(addr, (t as u128 * 100 + i as u128) * 1000);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(cache.len(), 1000);
    }
}
