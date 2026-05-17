//! Caching layer for AI model outputs
use redis::{Client, AsyncCommands};
use lru::LruCache;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::num::NonZeroUsize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::{info, warn, error, debug};
use std::time::{Duration, Instant};

/// Cache provider type
pub enum CacheProvider {
    /// In-memory LRU cache
    Memory,
    /// Redis cache
    Redis,
    /// Combined (layered) cache
    Layered,
}

/// Model cache for storing AI outputs
pub struct ModelCache {
    /// In-memory cache
    memory_cache: Arc<Mutex<LruCache<u64, CacheEntry>>>,
    /// Redis client if available
    redis_client: Option<Client>,
    /// Cache provider
    provider: CacheProvider,
    /// Cache hit statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache entry with expiration
struct CacheEntry {
    /// Output data
    data: Vec<u8>,
    /// Expiration timestamp
    expires_at: Instant,
}

/// Cache statistics
#[derive(Debug, Default)]
struct CacheStats {
    /// Number of gets
    gets: u64,
    /// Number of sets
    sets: u64,
    /// Number of memory hits
    memory_hits: u64,
    /// Number of redis hits
    redis_hits: u64,
    /// Number of misses
    misses: u64,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(provider: CacheProvider, memory_size: usize, redis_url: Option<String>) -> Self {
        // Create memory cache
        let memory_size = NonZeroUsize::new(memory_size).unwrap_or(NonZeroUsize::new(10000).unwrap());
        let memory_cache = Arc::new(Mutex::new(LruCache::new(memory_size)));
        
        // Create Redis client if URL provided
        let redis_client = if let Some(url) = redis_url {
            match Client::open(url) {
                Ok(client) => {
                    info!("Redis cache connected");
                    Some(client)
                },
                Err(e) => {
                    error!("Failed to connect to Redis: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Warn if Redis requested but not available
        if matches!(provider, CacheProvider::Redis | CacheProvider::Layered) && redis_client.is_none() {
            warn!("Redis cache requested but not available, falling back to memory cache");
        }
        
        Self {
            memory_cache,
            redis_client,
            provider,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    /// Get cached result
    pub async fn get(&self, model: &str, input: &[u8], ttl: u64) -> Option<Vec<u8>> {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.gets += 1;
        }
        
        // Try memory cache first
        if let Some(entry) = self.check_memory_cache(key).await {
            if entry.expires_at > Instant::now() {
                // Update stats
                {
                    let mut stats = self.stats.lock().await;
                    stats.memory_hits += 1;
                }
                return Some(entry.data);
            }
        }
        
        // If Redis is available and enabled, try it next
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                if let Some(data) = self.check_redis_cache(client, model, input).await {
                    // Also update memory cache for next time
                    self.update_memory_cache(key, &data, ttl).await;
                    
                    // Update stats
                    {
                        let mut stats = self.stats.lock().await;
                        stats.redis_hits += 1;
                    }
                    
                    return Some(data);
                }
            }
        }
        
        // Update stats for miss
        {
            let mut stats = self.stats.lock().await;
            stats.misses += 1;
        }
        
        None
    }
    
    /// Check memory cache
    async fn check_memory_cache(&self, key: u64) -> Option<CacheEntry> {
        let mut cache = self.memory_cache.lock().await;
        cache.get(&key).cloned()
    }
    
    /// Check Redis cache
    async fn check_redis_cache(&self, client: &Client, model: &str, input: &[u8]) -> Option<Vec<u8>> {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                match conn.get::<_, Option<Vec<u8>>>(&key).await {
                    Ok(Some(data)) => Some(data),
                    Ok(None) => None,
                    Err(e) => {
                        error!("Redis error while getting key {}: {}", key, e);
                        None
                    }
                }
            },
            Err(e) => {
                error!("Failed to get Redis connection: {}", e);
                None
            }
        }
    }
    
    /// Set value in cache
    pub async fn set(&self, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.sets += 1;
        }
        
        // Update memory cache
        self.update_memory_cache(key, output, ttl).await;
        
        // Update Redis if available
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                self.update_redis_cache(client, model, input, output, ttl).await;
            }
        }
    }
    
    /// Update memory cache
    async fn update_memory_cache(&self, key: u64, data: &[u8], ttl: u64) {
        let entry = CacheEntry {
            data: data.to_vec(),
            expires_at: Instant::now() + Duration::from_secs(ttl),
        };
        
        let mut cache = self.memory_cache.lock().await;
        cache.put(key, entry);
    }
    
    /// Update Redis cache
    async fn update_redis_cache(&self, client: &Client, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                let _: Result<(), redis::RedisError> = conn.set_ex(&key, output, ttl as usize).await;
            },
            Err(e) => {
                error!("Failed to get Redis connection for set: {}", e);
            }
        }
    }
    
    /// Generate cache key
    fn generate_key(model: &str, input: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        model.hash(&mut hasher);
        input.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash bytes
    fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.lock().await.clone()
    }
    
    /// Start periodic cleanup
    pub fn start_cleanup_task(&self) {
        let memory_cache = self.memory_cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut cache = memory_cache.lock().await;
                
                // Remove expired entries
                cache.retain(|_, entry| entry.expires_at > now);
                
                debug!("Cache cleanup completed, size: {}", cache.len());
            }
        });
    }
}
