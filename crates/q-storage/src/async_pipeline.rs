//! Reddio-Style Async Storage Pipeline for Q-NarwhalKnight
//!
//! v1.5.0-beta: Implementation based on Reddio paper (https://arxiv.org/abs/2503.04595)
//!
//! Key insight: Storage operations are 70% of execution overhead in blockchain systems.
//! This module implements three optimizations:
//!
//! 1. **Direct State Reading** - Hot balance cache bypasses RocksDB reads
//! 2. **Async Parallel Preloading** - Prefetch addresses from pending transactions
//! 3. **Pipelined Workflow** - Overlap read/execute/write phases for max throughput
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Async Storage Pipeline                               │
//! │                                                                         │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
//! │  │  Preloader  │───>│  Hot Cache  │───>│  Executor   │───>│  Writer  │ │
//! │  │  (prefetch) │    │  (bypass DB)│    │  (parallel) │    │  (batch) │ │
//! │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
//! │        │                   │                  │                 │      │
//! │        ▼                   ▼                  ▼                 ▼      │
//! │   Pending Txs        Cache Hits         State Deltas      RocksDB     │
//! │   (extract addrs)    (70% hit rate)    (validated)       (batched)   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::{Context, Result};
use dashmap::DashMap;
#[cfg(not(target_os = "windows"))]
use rocksdb::DB;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, info, trace, warn};

use q_types::Address;

/// Type alias for balance value
pub type Balance = u64;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the async storage pipeline
#[derive(Debug, Clone)]
pub struct AsyncPipelineConfig {
    /// Maximum entries in hot cache (LRU eviction)
    pub hot_cache_capacity: usize,
    /// Number of prefetch workers
    pub prefetch_workers: usize,
    /// Prefetch batch size
    pub prefetch_batch_size: usize,
    /// Maximum prefetch queue depth
    pub prefetch_queue_depth: usize,
    /// Enable speculative prefetching (from mempool)
    pub speculative_prefetch: bool,
    /// Cache TTL in seconds (0 = no expiry)
    pub cache_ttl_secs: u64,
    /// Pipeline parallelism (overlapping stages)
    pub pipeline_depth: usize,
}

impl Default for AsyncPipelineConfig {
    fn default() -> Self {
        Self {
            hot_cache_capacity: 100_000,     // 100K addresses (~8MB RAM)
            prefetch_workers: 4,              // 4 prefetch workers
            prefetch_batch_size: 256,         // Prefetch 256 addresses at a time
            prefetch_queue_depth: 10_000,     // Max pending prefetch requests
            speculative_prefetch: true,       // Prefetch from mempool
            cache_ttl_secs: 60,               // 60s TTL
            pipeline_depth: 3,                // 3 overlapping stages
        }
    }
}

// ============================================================================
// Hot Balance Cache (Direct State Reading)
// ============================================================================

/// Cached balance entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Balance value
    balance: Balance,
    /// When this entry was cached
    cached_at: Instant,
    /// Number of reads since caching
    read_count: u64,
    /// Dirty flag (modified but not persisted)
    dirty: bool,
}

/// LRU Hot Balance Cache
/// Provides O(1) balance lookups without RocksDB access
pub struct HotBalanceCache {
    /// Main cache storage (address -> balance)
    cache: DashMap<Address, CacheEntry>,
    /// Maximum capacity
    capacity: usize,
    /// Cache TTL
    ttl: Duration,
    /// Metrics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl HotBalanceCache {
    /// Create new hot balance cache
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            capacity,
            ttl: Duration::from_secs(ttl_secs),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Get balance from cache (returns None on miss)
    pub fn get(&self, address: &Address) -> Option<Balance> {
        if let Some(mut entry) = self.cache.get_mut(address) {
            // Check TTL
            if self.ttl.as_secs() > 0 && entry.cached_at.elapsed() > self.ttl {
                drop(entry);
                self.cache.remove(address);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            entry.read_count += 1;
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.balance)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert/update balance in cache
    pub fn insert(&self, address: Address, balance: Balance, dirty: bool) {
        // Evict if at capacity
        if self.cache.len() >= self.capacity {
            self.evict_lru();
        }

        self.cache.insert(address, CacheEntry {
            balance,
            cached_at: Instant::now(),
            read_count: 0,
            dirty,
        });
    }

    /// Mark entry as dirty (modified locally)
    pub fn mark_dirty(&self, address: &Address) {
        if let Some(mut entry) = self.cache.get_mut(address) {
            entry.dirty = true;
        }
    }

    /// Get all dirty entries (for persistence)
    pub fn get_dirty(&self) -> Vec<(Address, Balance)> {
        self.cache
            .iter()
            .filter(|e| e.dirty)
            .map(|e| (*e.key(), e.balance))
            .collect()
    }

    /// Clear dirty flags after persistence
    pub fn clear_dirty(&self) {
        for mut entry in self.cache.iter_mut() {
            entry.dirty = false;
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&self) {
        // Find entry with lowest read_count or oldest
        let mut to_evict: Option<Address> = None;
        let mut min_score = u64::MAX;

        for entry in self.cache.iter() {
            // Score = read_count * 1000 + (1000 - age_secs)
            // Lower score = more likely to evict
            let age_secs = entry.cached_at.elapsed().as_secs().min(1000);
            let score = entry.read_count.saturating_mul(1000) + (1000 - age_secs);

            if score < min_score {
                min_score = score;
                to_evict = Some(*entry.key());
            }
        }

        if let Some(addr) = to_evict {
            self.cache.remove(&addr);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Bulk insert (for prefetch results)
    pub fn bulk_insert(&self, entries: Vec<(Address, Balance)>) {
        for (addr, balance) in entries {
            self.insert(addr, balance, false);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> HotCacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        HotCacheStats {
            size: self.cache.len(),
            capacity: self.capacity,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }

    /// Clear entire cache
    pub fn clear(&self) {
        self.cache.clear();
    }
}

/// Hot cache statistics
#[derive(Debug, Clone)]
pub struct HotCacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

// ============================================================================
// Async Prefetcher (Parallel Node Loading)
// ============================================================================

/// Prefetch request
#[derive(Debug)]
struct PrefetchRequest {
    /// Addresses to prefetch
    addresses: Vec<Address>,
    /// Priority (higher = more urgent)
    priority: u8,
}

/// Async prefetcher - preloads balances before execution needs them
pub struct AsyncPrefetcher {
    /// Hot cache reference
    cache: Arc<HotBalanceCache>,
    /// RocksDB handle
    db: Arc<DB>,
    /// Balance column family name
    cf_balances: String,
    /// Request channel
    request_tx: mpsc::Sender<PrefetchRequest>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Worker handles
    _workers: Vec<tokio::task::JoinHandle<()>>,
    /// Metrics
    prefetched: AtomicU64,
    prefetch_time_us: AtomicU64,
}

impl AsyncPrefetcher {
    /// Create new async prefetcher
    pub fn new(
        cache: Arc<HotBalanceCache>,
        db: Arc<DB>,
        cf_balances: String,
        config: &AsyncPipelineConfig,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel(config.prefetch_queue_depth);
        let request_rx = Arc::new(RwLock::new(request_rx));
        let running = Arc::new(AtomicBool::new(true));

        let mut workers = Vec::with_capacity(config.prefetch_workers);

        for worker_id in 0..config.prefetch_workers {
            let cache_clone = cache.clone();
            let db_clone = db.clone();
            let cf_clone = cf_balances.clone();
            let rx_clone = request_rx.clone();
            let running_clone = running.clone();
            let batch_size = config.prefetch_batch_size;

            let handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker_id,
                    cache_clone,
                    db_clone,
                    cf_clone,
                    rx_clone,
                    running_clone,
                    batch_size,
                ).await;
            });

            workers.push(handle);
        }

        info!("🔮 AsyncPrefetcher started with {} workers", config.prefetch_workers);

        Self {
            cache,
            db,
            cf_balances,
            request_tx,
            running,
            _workers: workers,
            prefetched: AtomicU64::new(0),
            prefetch_time_us: AtomicU64::new(0),
        }
    }

    /// Worker loop - processes prefetch requests
    async fn worker_loop(
        worker_id: usize,
        cache: Arc<HotBalanceCache>,
        db: Arc<DB>,
        cf_balances: String,
        request_rx: Arc<RwLock<mpsc::Receiver<PrefetchRequest>>>,
        running: Arc<AtomicBool>,
        _batch_size: usize,
    ) {
        trace!("🔮 Prefetch worker {} started", worker_id);

        while running.load(Ordering::Relaxed) {
            // Try to get a request
            let request = {
                let mut rx = request_rx.write().await;
                match rx.try_recv() {
                    Ok(req) => Some(req),
                    Err(mpsc::error::TryRecvError::Empty) => None,
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            };

            if let Some(request) = request {
                // Filter out addresses already in cache
                let addresses_to_fetch: Vec<_> = request.addresses
                    .into_iter()
                    .filter(|addr| cache.get(addr).is_none())
                    .collect();

                if !addresses_to_fetch.is_empty() {
                    // Fetch from RocksDB (blocking - use spawn_blocking)
                    let db_clone = db.clone();
                    let cf_clone = cf_balances.clone();
                    let cache_clone = cache.clone();

                    let _ = tokio::task::spawn_blocking(move || {
                        Self::fetch_from_db(&db_clone, &cf_clone, &cache_clone, addresses_to_fetch);
                    }).await;
                }
            } else {
                // No request - brief sleep
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        }

        trace!("🔮 Prefetch worker {} stopped", worker_id);
    }

    /// Fetch balances from RocksDB and insert into cache
    fn fetch_from_db(
        db: &Arc<DB>,
        cf_balances: &str,
        cache: &HotBalanceCache,
        addresses: Vec<Address>,
    ) {
        let start = Instant::now();

        if let Some(cf) = db.cf_handle(cf_balances) {
            let mut results = Vec::with_capacity(addresses.len());

            for addr in &addresses {
                match db.get_cf(&cf, addr) {
                    Ok(Some(data)) if data.len() >= 8 => {
                        let balance = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]));
                        results.push((*addr, balance));
                    }
                    Ok(_) => {
                        // Not found or invalid - cache as 0
                        results.push((*addr, 0));
                    }
                    Err(e) => {
                        trace!("Prefetch error for {:?}: {}", addr, e);
                    }
                }
            }

            // Bulk insert into cache
            cache.bulk_insert(results);

            let elapsed = start.elapsed();
            trace!(
                "🔮 Prefetched {} addresses in {:?}",
                addresses.len(),
                elapsed
            );
        }
    }

    /// Request prefetch for addresses (non-blocking)
    pub fn prefetch(&self, addresses: Vec<Address>, priority: u8) {
        if addresses.is_empty() {
            return;
        }

        let _ = self.request_tx.try_send(PrefetchRequest { addresses, priority });
        self.prefetched.fetch_add(1, Ordering::Relaxed);
    }

    /// Extract addresses from transactions and prefetch
    pub fn prefetch_from_transactions(&self, transactions: &[q_types::Transaction]) {
        let mut addresses = HashSet::new();

        for tx in transactions {
            addresses.insert(tx.from);
            addresses.insert(tx.to);
        }

        let addr_vec: Vec<_> = addresses.into_iter().collect();
        self.prefetch(addr_vec, 5); // Medium priority
    }

    /// Get prefetch statistics
    pub fn stats(&self) -> PrefetchStats {
        let total = self.prefetched.load(Ordering::Relaxed);
        let time_us = self.prefetch_time_us.load(Ordering::Relaxed);

        PrefetchStats {
            total_requests: total,
            avg_time_us: if total > 0 { time_us / total } else { 0 },
        }
    }

    /// Shutdown prefetcher
    pub async fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Prefetch statistics
#[derive(Debug, Clone)]
pub struct PrefetchStats {
    pub total_requests: u64,
    pub avg_time_us: u64,
}

// ============================================================================
// Pipelined Workflow
// ============================================================================

/// Pipeline stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Reading state from cache/DB
    Read,
    /// Executing transactions
    Execute,
    /// Writing state updates
    Write,
}

/// Pipeline task
#[derive(Debug, Clone)]
pub struct PipelineTask {
    /// Block height being processed
    pub height: u64,
    /// Current stage
    pub stage: PipelineStage,
    /// Addresses needed (for prefetch)
    pub addresses: Vec<Address>,
    /// State deltas to write
    pub deltas: HashMap<Address, i64>,
    /// Started at
    pub started_at: Instant,
}

/// Pipelined workflow manager
/// Overlaps read/execute/write for maximum throughput
pub struct PipelineManager {
    /// Active tasks in each stage
    read_queue: Arc<RwLock<VecDeque<PipelineTask>>>,
    execute_queue: Arc<RwLock<VecDeque<PipelineTask>>>,
    write_queue: Arc<RwLock<VecDeque<PipelineTask>>>,
    /// Maximum tasks per stage
    max_depth: usize,
    /// Semaphores for flow control
    read_sem: Arc<Semaphore>,
    execute_sem: Arc<Semaphore>,
    write_sem: Arc<Semaphore>,
    /// Metrics
    completed_tasks: AtomicU64,
    total_latency_us: AtomicU64,
}

impl PipelineManager {
    /// Create new pipeline manager
    pub fn new(depth: usize) -> Self {
        Self {
            read_queue: Arc::new(RwLock::new(VecDeque::with_capacity(depth))),
            execute_queue: Arc::new(RwLock::new(VecDeque::with_capacity(depth))),
            write_queue: Arc::new(RwLock::new(VecDeque::with_capacity(depth))),
            max_depth: depth,
            read_sem: Arc::new(Semaphore::new(depth)),
            execute_sem: Arc::new(Semaphore::new(depth)),
            write_sem: Arc::new(Semaphore::new(depth)),
            completed_tasks: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }

    /// Submit task to read stage
    pub async fn submit_read(&self, task: PipelineTask) -> Result<()> {
        let _permit = self.read_sem.acquire().await?;

        let mut queue = self.read_queue.write().await;
        if queue.len() >= self.max_depth {
            return Err(anyhow::anyhow!("Read queue full"));
        }
        queue.push_back(task);

        Ok(())
    }

    /// Move task from read to execute stage
    pub async fn promote_to_execute(&self) -> Option<PipelineTask> {
        let mut read_queue = self.read_queue.write().await;
        if let Some(mut task) = read_queue.pop_front() {
            drop(read_queue);

            // Try to acquire execute permit
            if let Ok(_permit) = self.execute_sem.try_acquire() {
                task.stage = PipelineStage::Execute;

                let mut exec_queue = self.execute_queue.write().await;
                exec_queue.push_back(task.clone());

                return Some(task);
            } else {
                // Put back in read queue
                let mut read_queue = self.read_queue.write().await;
                read_queue.push_front(task);
            }
        }
        None
    }

    /// Move task from execute to write stage
    pub async fn promote_to_write(&self, height: u64, deltas: HashMap<Address, i64>) -> Result<()> {
        let mut exec_queue = self.execute_queue.write().await;

        // Find and remove the task
        let mut found_idx = None;
        for (idx, task) in exec_queue.iter().enumerate() {
            if task.height == height {
                found_idx = Some(idx);
                break;
            }
        }

        if let Some(idx) = found_idx {
            let mut task = exec_queue.remove(idx).unwrap();
            drop(exec_queue);

            // Try to acquire write permit
            let _permit = self.write_sem.acquire().await?;

            task.stage = PipelineStage::Write;
            task.deltas = deltas;

            let mut write_queue = self.write_queue.write().await;
            write_queue.push_back(task);

            Ok(())
        } else {
            Err(anyhow::anyhow!("Task not found in execute queue"))
        }
    }

    /// Complete write stage and finalize task
    pub async fn complete_write(&self, height: u64) -> Result<Duration> {
        let mut write_queue = self.write_queue.write().await;

        // Find and remove the task
        let mut found_idx = None;
        for (idx, task) in write_queue.iter().enumerate() {
            if task.height == height {
                found_idx = Some(idx);
                break;
            }
        }

        if let Some(idx) = found_idx {
            let task = write_queue.remove(idx).unwrap();
            let latency = task.started_at.elapsed();

            self.completed_tasks.fetch_add(1, Ordering::Relaxed);
            self.total_latency_us.fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

            Ok(latency)
        } else {
            Err(anyhow::anyhow!("Task not found in write queue"))
        }
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        let completed = self.completed_tasks.load(Ordering::Relaxed);
        let total_us = self.total_latency_us.load(Ordering::Relaxed);

        PipelineStats {
            completed_tasks: completed,
            avg_latency_us: if completed > 0 { total_us / completed } else { 0 },
            max_depth: self.max_depth,
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub completed_tasks: u64,
    pub avg_latency_us: u64,
    pub max_depth: usize,
}

// ============================================================================
// Main Async Storage Pipeline
// ============================================================================

/// Main async storage pipeline combining all optimizations
pub struct AsyncStoragePipeline {
    /// Hot balance cache
    pub cache: Arc<HotBalanceCache>,
    /// Async prefetcher
    pub prefetcher: Arc<AsyncPrefetcher>,
    /// Pipeline manager
    pub pipeline: Arc<PipelineManager>,
    /// RocksDB handle
    db: Arc<DB>,
    /// Column family for balances
    cf_balances: String,
    /// Configuration
    config: AsyncPipelineConfig,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Cumulative metrics
    total_reads: AtomicU64,
    total_writes: AtomicU64,
    cache_bypass_reads: AtomicU64,
}

impl AsyncStoragePipeline {
    /// Create new async storage pipeline
    pub fn new(db: Arc<DB>, cf_balances: String, config: AsyncPipelineConfig) -> Result<Self> {
        let cache = Arc::new(HotBalanceCache::new(
            config.hot_cache_capacity,
            config.cache_ttl_secs,
        ));

        let prefetcher = Arc::new(AsyncPrefetcher::new(
            cache.clone(),
            db.clone(),
            cf_balances.clone(),
            &config,
        ));

        let pipeline = Arc::new(PipelineManager::new(config.pipeline_depth));

        info!("🚀 AsyncStoragePipeline initialized (Reddio-style)");
        info!("   Hot cache capacity: {}", config.hot_cache_capacity);
        info!("   Prefetch workers: {}", config.prefetch_workers);
        info!("   Pipeline depth: {}", config.pipeline_depth);

        Ok(Self {
            cache,
            prefetcher,
            pipeline,
            db,
            cf_balances,
            config,
            running: Arc::new(AtomicBool::new(true)),
            total_reads: AtomicU64::new(0),
            total_writes: AtomicU64::new(0),
            cache_bypass_reads: AtomicU64::new(0),
        })
    }

    /// Get balance with hot cache (direct state reading)
    pub fn get_balance(&self, address: &Address) -> Result<Balance> {
        self.total_reads.fetch_add(1, Ordering::Relaxed);

        // Try hot cache first (O(1) lookup)
        if let Some(balance) = self.cache.get(address) {
            return Ok(balance);
        }

        // Cache miss - read from RocksDB
        self.cache_bypass_reads.fetch_add(1, Ordering::Relaxed);

        let cf = self.db.cf_handle(&self.cf_balances)
            .context("Balance column family not found")?;

        match self.db.get_cf(&cf, address)? {
            Some(data) if data.len() >= 8 => {
                let balance = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]));
                // Insert into cache for next time
                self.cache.insert(*address, balance, false);
                Ok(balance)
            }
            _ => {
                // Not found - cache as 0
                self.cache.insert(*address, 0, false);
                Ok(0)
            }
        }
    }

    /// Get multiple balances (batch operation)
    pub fn get_balances(&self, addresses: &[Address]) -> HashMap<Address, Balance> {
        let mut results = HashMap::with_capacity(addresses.len());

        for addr in addresses {
            if let Ok(balance) = self.get_balance(addr) {
                results.insert(*addr, balance);
            }
        }

        results
    }

    /// Update balance (writes to cache first, batched to DB later)
    pub fn update_balance(&self, address: Address, new_balance: Balance) {
        self.total_writes.fetch_add(1, Ordering::Relaxed);
        self.cache.insert(address, new_balance, true); // Mark as dirty
    }

    /// Apply balance deltas (optimized for transaction processing)
    pub fn apply_deltas(&self, deltas: &HashMap<Address, i64>) -> Result<()> {
        for (addr, delta) in deltas {
            let current = self.get_balance(addr)?;
            let new_balance = if *delta >= 0 {
                current.saturating_add(*delta as u64)
            } else {
                current.saturating_sub((-*delta) as u64)
            };
            self.update_balance(*addr, new_balance);
        }
        Ok(())
    }

    /// Prefetch addresses for upcoming transactions
    pub fn prefetch_for_block(&self, transactions: &[q_types::Transaction]) {
        if self.config.speculative_prefetch {
            self.prefetcher.prefetch_from_transactions(transactions);
        }
    }

    /// Flush dirty entries to RocksDB
    pub fn flush_to_db(&self) -> Result<usize> {
        let dirty = self.cache.get_dirty();
        if dirty.is_empty() {
            return Ok(0);
        }

        let cf = self.db.cf_handle(&self.cf_balances)
            .context("Balance column family not found")?;

        let mut batch = rocksdb::WriteBatch::default();

        for (addr, balance) in &dirty {
            batch.put_cf(&cf, addr, balance.to_le_bytes());
        }

        let mut opts = rocksdb::WriteOptions::default();
        opts.set_sync(false); // Async for speed, fsync periodically

        self.db.write_opt(batch, &opts)?;
        self.cache.clear_dirty();

        debug!("💾 Flushed {} dirty entries to RocksDB", dirty.len());

        Ok(dirty.len())
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> AsyncPipelineStats {
        let total_reads = self.total_reads.load(Ordering::Relaxed);
        let cache_bypasses = self.cache_bypass_reads.load(Ordering::Relaxed);

        AsyncPipelineStats {
            cache: self.cache.stats(),
            prefetch: self.prefetcher.stats(),
            pipeline: self.pipeline.stats(),
            total_reads,
            total_writes: self.total_writes.load(Ordering::Relaxed),
            direct_read_rate: if total_reads > 0 {
                1.0 - (cache_bypasses as f64 / total_reads as f64)
            } else {
                0.0
            },
        }
    }

    /// Log statistics summary
    pub fn log_stats(&self) {
        let stats = self.stats();
        info!("📊 AsyncStoragePipeline Statistics:");
        info!("   🔥 Hot Cache: {}/{} entries, {:.1}% hit rate",
            stats.cache.size, stats.cache.capacity, stats.cache.hit_rate * 100.0);
        info!("   🔮 Prefetch: {} requests, avg {}μs",
            stats.prefetch.total_requests, stats.prefetch.avg_time_us);
        info!("   🔄 Pipeline: {} tasks, avg {}μs latency",
            stats.pipeline.completed_tasks, stats.pipeline.avg_latency_us);
        info!("   📈 Direct Read Rate: {:.1}% (target: 70%+)",
            stats.direct_read_rate * 100.0);
    }

    /// Shutdown pipeline
    pub async fn shutdown(&self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);

        // Flush remaining dirty entries
        let flushed = self.flush_to_db()?;
        info!("🛑 AsyncStoragePipeline shutdown (flushed {} entries)", flushed);

        self.prefetcher.shutdown().await;

        Ok(())
    }
}

/// Comprehensive pipeline statistics
#[derive(Debug, Clone)]
pub struct AsyncPipelineStats {
    pub cache: HotCacheStats,
    pub prefetch: PrefetchStats,
    pub pipeline: PipelineStats,
    pub total_reads: u64,
    pub total_writes: u64,
    /// Percentage of reads served from cache (target: 70%+)
    pub direct_read_rate: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_cache_basic() {
        let cache = HotBalanceCache::new(100, 60);
        let addr = [1u8; 32];

        // Miss
        assert!(cache.get(&addr).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Insert
        cache.insert(addr, 1000, false);

        // Hit
        assert_eq!(cache.get(&addr), Some(1000));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_hot_cache_lru_eviction() {
        let cache = HotBalanceCache::new(3, 0);

        // Fill cache
        cache.insert([1u8; 32], 100, false);
        cache.insert([2u8; 32], 200, false);
        cache.insert([3u8; 32], 300, false);

        // Read one to increase its score
        cache.get(&[2u8; 32]);
        cache.get(&[2u8; 32]);

        // Insert fourth - should evict lowest score
        cache.insert([4u8; 32], 400, false);

        // [2] should survive (highest read count)
        assert_eq!(cache.get(&[2u8; 32]), Some(200));
        // [4] should exist
        assert_eq!(cache.get(&[4u8; 32]), Some(400));

        // One of [1], [3] should be evicted
        let evicted = cache.get(&[1u8; 32]).is_none() || cache.get(&[3u8; 32]).is_none();
        assert!(evicted);
    }

    #[test]
    fn test_hot_cache_dirty_tracking() {
        let cache = HotBalanceCache::new(100, 0);
        let addr1 = [1u8; 32];
        let addr2 = [2u8; 32];

        cache.insert(addr1, 100, true);  // dirty
        cache.insert(addr2, 200, false); // clean

        let dirty = cache.get_dirty();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], (addr1, 100));

        cache.clear_dirty();
        assert!(cache.get_dirty().is_empty());
    }

    #[test]
    fn test_pipeline_manager_flow() {
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            let pipeline = PipelineManager::new(3);

            // Submit to read
            let task = PipelineTask {
                height: 1,
                stage: PipelineStage::Read,
                addresses: vec![[1u8; 32]],
                deltas: HashMap::new(),
                started_at: Instant::now(),
            };

            pipeline.submit_read(task).await.unwrap();

            // Promote to execute
            let task = pipeline.promote_to_execute().await;
            assert!(task.is_some());
            assert_eq!(task.unwrap().stage, PipelineStage::Execute);

            // Promote to write
            let mut deltas = HashMap::new();
            deltas.insert([1u8; 32], 100i64);
            pipeline.promote_to_write(1, deltas).await.unwrap();

            // Complete write
            let latency = pipeline.complete_write(1).await.unwrap();
            assert!(latency.as_micros() > 0);

            let stats = pipeline.stats();
            assert_eq!(stats.completed_tasks, 1);
        });
    }
}
