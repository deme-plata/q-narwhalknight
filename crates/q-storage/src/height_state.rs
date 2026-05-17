/// Height State Cache - Eliminates Binary Search Storms
///
/// This module provides a cached height state to prevent repeated expensive
/// binary search operations during normal operation and especially during shutdown.
///
/// Key features:
/// - Atomic cached height value (lock-free reads)
/// - Time-based cache freshness tracking
/// - Shutdown mode for fast pointer-only reads
/// - Watch channel for height updates (observers can subscribe)

use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{watch, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Height state with caching and shutdown mode
#[derive(Clone)]
pub struct HeightState {
    /// Cached height value (atomic for lock-free reads)
    cached: Arc<AtomicU64>,

    /// Last time the cache was refreshed
    last_refresh: Arc<RwLock<Instant>>,

    /// Shutdown flag (when true, skip binary search and use pointer only)
    shutdown: Arc<AtomicBool>,

    /// Watch channel sender for broadcasting height updates
    pub tx: watch::Sender<u64>,

    /// Watch channel receiver for subscribing to height updates
    pub rx: watch::Receiver<u64>,
}

impl HeightState {
    /// Create new height state with initial height
    pub fn new(initial: u64) -> Self {
        let (tx, rx) = watch::channel(initial);
        Self {
            cached: Arc::new(AtomicU64::new(initial)),
            // Set last refresh to long ago to force initial refresh
            // Use checked_sub to avoid panic on Windows where Instant is based on uptime
            last_refresh: Arc::new(RwLock::new(Instant::now().checked_sub(Duration::from_secs(3600)).unwrap_or(Instant::now()))),
            shutdown: Arc::new(AtomicBool::new(false)),
            tx,
            rx,
        }
    }

    /// Mark that the system is shutting down (enables fast shutdown mode)
    pub fn mark_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Get the cached height value (lock-free)
    pub fn cached(&self) -> u64 {
        self.cached.load(Ordering::Relaxed)
    }

    /// Check if the system is in shutdown mode
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Update the cached height and broadcast to subscribers
    /// 🚨 v1.0.41-beta: CRITICAL FIX - Only update if new height is HIGHER
    /// This prevents height regression when out-of-order batch responses arrive
    /// (e.g., batch 1-200 arriving after batch 800-1000 shouldn't set cache to 200)
    pub async fn update(&self, h: u64) {
        let current = self.cached.load(Ordering::Relaxed);
        if h > current {
            self.cached.store(h, Ordering::Relaxed);
            *self.last_refresh.write().await = Instant::now();
            let _ = self.tx.send(h);
        }
        // If h <= current, silently ignore (not an error, just out-of-order)
    }

    /// 🛡️ v7.2.3: Force-set the height cache during startup recovery.
    /// Unlike update(), this allows DOWNWARD correction when the database scan
    /// finds that actual contiguous height < cached height (e.g., after crash
    /// where turbo_sync blocks were lost from OS page cache).
    pub async fn force_set(&self, h: u64) {
        let current = self.cached.load(Ordering::Relaxed);
        if h != current {
            tracing::warn!(
                "🛡️ [HEIGHT CACHE v7.2.3] Force-setting height cache: {} → {} (delta: {})",
                current, h, h as i64 - current as i64
            );
            self.cached.store(h, Ordering::Relaxed);
            *self.last_refresh.write().await = Instant::now();
            let _ = self.tx.send(h);
        }
    }

    /// Check if the cache is fresh (within max_age)
    pub async fn is_cache_fresh(&self, max_age: Duration) -> bool {
        self.last_refresh.read().await.elapsed() < max_age
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_height_state_creation() {
        let state = HeightState::new(100);
        assert_eq!(state.cached(), 100);
        assert!(!state.is_shutdown());
    }

    #[tokio::test]
    async fn test_height_state_update() {
        let state = HeightState::new(0);
        state.update(42).await;
        assert_eq!(state.cached(), 42);
    }

    #[tokio::test]
    async fn test_cache_freshness() {
        let state = HeightState::new(0);

        // Initially stale (set to 1 hour ago)
        assert!(!state.is_cache_fresh(Duration::from_secs(5)).await);

        // Update makes it fresh
        state.update(100).await;
        assert!(state.is_cache_fresh(Duration::from_secs(5)).await);

        // After delay, becomes stale
        sleep(Duration::from_secs(6)).await;
        assert!(!state.is_cache_fresh(Duration::from_secs(5)).await);
    }

    #[tokio::test]
    async fn test_shutdown_mode() {
        let state = HeightState::new(0);
        assert!(!state.is_shutdown());

        state.mark_shutdown();
        assert!(state.is_shutdown());
    }

    #[tokio::test]
    async fn test_watch_channel_subscription() {
        let state = HeightState::new(0);
        let mut rx = state.tx.subscribe();

        state.update(42).await;

        assert_eq!(*rx.borrow_and_update(), 42);
    }
}
