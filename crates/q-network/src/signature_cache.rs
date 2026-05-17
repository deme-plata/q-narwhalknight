/// Signature verification cache with TOCTOU fix
///
/// This module implements Showstopper #3 fix: Signature cache with public key fingerprinting
///
/// Performance Impact:
/// - Without cache: 50-100μs per verification × 1000 msg/s = 50-100ms CPU/s
/// - With cache (80% hit rate): 10-20ms CPU/s (4-5× improvement)
///
/// Security Features:
/// - Cache key includes public key fingerprint (prevents cache poisoning on key rotation)
/// - TOCTOU fix: Re-check after acquiring write lock
/// - LRU eviction at 10,000 entries
/// - 60-second TTL

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::debug;

/// Cache key includes both message ID and public key fingerprint
///
/// This prevents cache poisoning attacks where:
/// 1. Attacker compromises NodeA's key
/// 2. NodeA rotates key and announces new public key
/// 3. Attacker sends messages with OLD key
/// 4. Without fingerprint, cache would serve OLD verification results
///
/// With fingerprint:
/// - Old key has different fingerprint → cache miss → verification fails
/// - New key has different fingerprint → cache miss → verification succeeds
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct CacheKey {
    message_id: String,
    public_key_fingerprint: [u8; 32], // SHA3-256 of public key
}

/// Signature verification cache with TOCTOU fix
///
/// Thread Safety:
/// - Uses RwLock for concurrent reads, exclusive writes
/// - TOCTOU fix: Re-checks cache after acquiring write lock
///
/// Performance:
/// - Read lock: Multiple threads can check cache simultaneously
/// - Write lock: Exclusive for cache updates
/// - LRU eviction: O(n) scan (acceptable at 10k entries)
pub struct SignatureCache {
    /// Cache storage: key → (verification result, cached timestamp)
    cache: Arc<RwLock<HashMap<CacheKey, (bool, Instant)>>>,

    /// Time-to-live for cached entries (60 seconds)
    ttl: Duration,

    /// Maximum cache size before LRU eviction
    max_entries: usize,
}

impl SignatureCache {
    /// Create new signature cache
    ///
    /// Configuration:
    /// - TTL: 60 seconds (balances freshness vs. cache hit rate)
    /// - Max entries: 10,000 (prevents unbounded memory growth)
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl: Duration::from_secs(60),
            max_entries: 10_000,
        }
    }

    /// Create cache with custom configuration (for testing)
    pub fn new_with_config(ttl: Duration, max_entries: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            max_entries,
        }
    }

    /// Check cache and verify signature if cache miss
    ///
    /// CRITICAL: Includes TOCTOU (Time-Of-Check-Time-Of-Use) fix
    ///
    /// TOCTOU Race Condition:
    /// 1. Thread A: Read lock, cache miss, drop read lock
    /// 2. Thread B: Write lock, verify signature, cache result, drop write lock
    /// 3. Thread A: Write lock, verify signature AGAIN (wasted work)
    ///
    /// Fix: Re-check cache after acquiring write lock (Thread A sees Thread B's result)
    pub async fn check_and_cache<F, Fut>(
        &self,
        message_id: &str,
        public_key_fingerprint: [u8; 32],
        verify_fn: F,
    ) -> bool
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let cache_key = CacheKey {
            message_id: message_id.to_string(),
            public_key_fingerprint,
        };

        // Step 1: Try to read from cache (fast path)
        {
            let cache = self.cache.read().await;
            if let Some((valid, cached_at)) = cache.get(&cache_key) {
                if cached_at.elapsed() < self.ttl {
                    debug!("✅ Signature cache HIT for {}", message_id);
                    debug!("   Fingerprint: {}", hex::encode(&public_key_fingerprint[..8]));
                    return *valid;
                } else {
                    debug!("⏰ Signature cache EXPIRED for {} (age: {}s)",
                           message_id, cached_at.elapsed().as_secs());
                }
            }
        } // Read lock dropped here

        // Step 2: Cache miss - acquire write lock
        let mut cache = self.cache.write().await;

        // Step 3: CRITICAL - Re-check after acquiring write lock (TOCTOU fix)
        // Another thread may have verified while we were waiting for write lock
        if let Some((valid, cached_at)) = cache.get(&cache_key) {
            if cached_at.elapsed() < self.ttl {
                debug!("✅ Cache HIT after write lock (another thread verified)");
                debug!("   Message: {}", message_id);
                return *valid;
            }
        }

        // Step 4: Actually verify (expensive operation)
        debug!("💾 Signature cache MISS for {}", message_id);
        debug!("   Fingerprint: {}", hex::encode(&public_key_fingerprint[..8]));

        let start = Instant::now();
        let valid = verify_fn().await;
        let duration = start.elapsed();

        debug!("🔐 Signature verification completed: {} ({}μs)",
               if valid { "VALID" } else { "INVALID" },
               duration.as_micros());

        // Step 5: LRU eviction if cache too large
        if cache.len() >= self.max_entries {
            self.evict_oldest(&mut cache);
        }

        // Step 6: Cache result
        cache.insert(cache_key, (valid, Instant::now()));

        valid
    }

    /// Evict oldest entry (simple LRU)
    ///
    /// Performance: O(n) scan (acceptable at 10k entries)
    /// Alternative: Use lru crate for O(1) eviction if needed
    fn evict_oldest(&self, cache: &mut HashMap<CacheKey, (bool, Instant)>) {
        if let Some(oldest_key) = cache
            .iter()
            .min_by_key(|(_, (_, cached_at))| *cached_at)
            .map(|(k, _)| k.clone())
        {
            cache.remove(&oldest_key);
            debug!("🗑️  Evicted oldest cache entry (cache size: {})", cache.len());
        }
    }

    /// Invalidate all entries for a specific public key
    ///
    /// Use case: When a node rotates its key, invalidate all cached
    /// verification results for the old key
    pub async fn invalidate_for_public_key(&self, public_key_fingerprint: [u8; 32]) {
        let mut cache = self.cache.write().await;
        let initial_size = cache.len();

        cache.retain(|key, _| key.public_key_fingerprint != public_key_fingerprint);

        let removed = initial_size - cache.len();
        if removed > 0 {
            debug!("🔄 Invalidated {} cache entries for rotated key", removed);
            debug!("   Fingerprint: {}", hex::encode(&public_key_fingerprint[..8]));
        }
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total_entries = cache.len();

        // Count valid vs invalid entries
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut expired_count = 0;

        for (valid, cached_at) in cache.values() {
            if cached_at.elapsed() >= self.ttl {
                expired_count += 1;
            } else if *valid {
                valid_count += 1;
            } else {
                invalid_count += 1;
            }
        }

        CacheStats {
            total_entries,
            valid_count,
            invalid_count,
            expired_count,
            max_entries: self.max_entries,
            ttl_seconds: self.ttl.as_secs(),
        }
    }

    /// Clear entire cache (for testing or emergency)
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let cleared = cache.len();
        cache.clear();
        debug!("🗑️  Cleared entire signature cache ({} entries)", cleared);
    }
}

impl Default for SignatureCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub valid_count: usize,
    pub invalid_count: usize,
    pub expired_count: usize,
    pub max_entries: usize,
    pub ttl_seconds: u64,
}

impl CacheStats {
    /// Calculate cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        self.total_entries as f64 / self.max_entries as f64
    }

    /// Calculate percentage of valid entries (0.0 to 1.0)
    pub fn valid_percentage(&self) -> f64 {
        if self.total_entries == 0 {
            return 0.0;
        }
        self.valid_count as f64 / self.total_entries as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = SignatureCache::new();
        let message_id = "test_message_1";
        let fingerprint = [1u8; 32];

        // First verification - cache miss
        let mut call_count = 0;
        let result1 = cache.check_and_cache(message_id, fingerprint, || async {
            call_count += 1;
            true
        }).await;

        assert!(result1);
        assert_eq!(call_count, 1);

        // Second verification - cache hit (should NOT call verify_fn)
        let result2 = cache.check_and_cache(message_id, fingerprint, || async {
            call_count += 1;
            true
        }).await;

        assert!(result2);
        assert_eq!(call_count, 1); // Still 1 - verify_fn not called
    }

    #[tokio::test]
    async fn test_cache_miss_different_fingerprint() {
        let cache = SignatureCache::new();
        let message_id = "test_message_1";
        let fingerprint1 = [1u8; 32];
        let fingerprint2 = [2u8; 32];

        // First verification with fingerprint1
        let mut call_count = 0;
        let result1 = cache.check_and_cache(message_id, fingerprint1, || async {
            call_count += 1;
            true
        }).await;

        assert!(result1);
        assert_eq!(call_count, 1);

        // Second verification with DIFFERENT fingerprint - cache miss
        let result2 = cache.check_and_cache(message_id, fingerprint2, || async {
            call_count += 1;
            true
        }).await;

        assert!(result2);
        assert_eq!(call_count, 2); // verify_fn called again
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = SignatureCache::new_with_config(
            Duration::from_millis(100), // 100ms TTL
            1000,
        );

        let message_id = "test_message_1";
        let fingerprint = [1u8; 32];

        // First verification
        let mut call_count = 0;
        let result1 = cache.check_and_cache(message_id, fingerprint, || async {
            call_count += 1;
            true
        }).await;

        assert!(result1);
        assert_eq!(call_count, 1);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Second verification after expiration - cache miss
        let result2 = cache.check_and_cache(message_id, fingerprint, || async {
            call_count += 1;
            true
        }).await;

        assert!(result2);
        assert_eq!(call_count, 2); // verify_fn called again
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = SignatureCache::new_with_config(
            Duration::from_secs(60),
            3, // Max 3 entries
        );

        let fingerprint = [1u8; 32];

        // Add 3 entries
        for i in 0..3 {
            let message_id = format!("message_{}", i);
            cache.check_and_cache(&message_id, fingerprint, || async { true }).await;
        }

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 3);

        // Add 4th entry - should evict oldest
        cache.check_and_cache("message_3", fingerprint, || async { true }).await;

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 3); // Still 3 (evicted oldest)
    }

    #[tokio::test]
    async fn test_invalidate_for_public_key() {
        let cache = SignatureCache::new();
        let fingerprint1 = [1u8; 32];
        let fingerprint2 = [2u8; 32];

        // Cache results for two different keys
        cache.check_and_cache("msg1", fingerprint1, || async { true }).await;
        cache.check_and_cache("msg2", fingerprint1, || async { true }).await;
        cache.check_and_cache("msg3", fingerprint2, || async { true }).await;

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 3);

        // Invalidate fingerprint1
        cache.invalidate_for_public_key(fingerprint1).await;

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 1); // Only fingerprint2 entry remains
    }

    #[tokio::test]
    async fn test_toctou_fix() {
        use std::sync::Arc;

        let cache = Arc::new(SignatureCache::new());
        let message_id = "test_message";
        let fingerprint = [1u8; 32];

        // Simulate race condition:
        // Thread 1 and Thread 2 both try to verify same message simultaneously

        let cache1 = cache.clone();
        let cache2 = cache.clone();

        let mut verify_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let verify_count1 = verify_count.clone();
        let verify_count2 = verify_count.clone();

        // Start both threads
        let handle1 = tokio::spawn(async move {
            cache1.check_and_cache(message_id, fingerprint, || async {
                verify_count1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await; // Simulate slow verification
                true
            }).await
        });

        let handle2 = tokio::spawn(async move {
            cache2.check_and_cache(message_id, fingerprint, || async {
                verify_count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                true
            }).await
        });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        assert!(result1);
        assert!(result2);

        // TOCTOU fix ensures verify_fn called at most ONCE
        // (Second thread sees first thread's cached result)
        let final_count = verify_count.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(final_count, 1, "TOCTOU fix failed - verify_fn called {} times", final_count);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = SignatureCache::new();
        let fingerprint = [1u8; 32];

        // Add valid and invalid entries
        cache.check_and_cache("msg1", fingerprint, || async { true }).await;
        cache.check_and_cache("msg2", fingerprint, || async { false }).await;
        cache.check_and_cache("msg3", fingerprint, || async { true }).await;

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.valid_count, 2);
        assert_eq!(stats.invalid_count, 1);
        assert_eq!(stats.expired_count, 0);
        assert!((stats.valid_percentage() - 0.666).abs() < 0.01);
    }
}
