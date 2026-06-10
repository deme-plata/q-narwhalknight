//! Sophisticated Gossipsub Message Queue System
//!
//! v3.4.13-beta: Priority-based message queue with rate limiting and backpressure
//! to prevent gossipsub send queue overflow during high-throughput periods.
//!
//! Features:
//! - Priority queuing: Blocks > Peer Heights > Balance Updates > Mining Solutions
//! - Per-message-type rate limiting
//! - Queue overflow protection with graceful degradation
//! - Backpressure signaling
//! - Metrics for monitoring queue health

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, trace};
use lazy_static::lazy_static;

/// Message priority levels (higher = more important)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Critical: Block propagation (consensus critical)
    Critical = 100,
    /// High: Peer height announcements (sync coordination)
    High = 75,
    /// Normal: Balance updates (user experience)
    Normal = 50,
    /// Low: Mining solutions (can be dropped under load)
    Low = 25,
    /// Lowest: Telemetry, metrics (always droppable)
    Lowest = 10,
}

impl MessagePriority {
    /// Get priority from topic name
    /// v3.5.23-beta: Transactions now Critical priority for faster P2P propagation
    pub fn from_topic(topic: &str) -> Self {
        if topic.contains("/blocks") || topic.contains("/turbo-sync") {
            MessagePriority::Critical
        } else if topic.contains("/transactions") || topic.contains("/mempool-txs") {
            // v3.5.23-beta: Elevate transactions to Critical priority
            // This removes rate limiting and ensures immediate propagation
            // Previously transactions went through Normal priority with 50ms rate limit
            MessagePriority::Critical
        } else if topic.contains("/peer-heights") || topic.contains("/height-proofs") {
            MessagePriority::High
        } else if topic.contains("/balance") || topic.contains("/wallets") {
            MessagePriority::Normal
        } else if topic.contains("/state-sync") {
            // v5.3.0: State sync requests/responses - High priority (no rate limit)
            // State sync is critical for new/restarted nodes to get contracts, pools, balances
            MessagePriority::High
        } else if topic.contains("/bridge-attestations") {
            // v7.3.1: Bridge attestation messages - Critical priority
            // Multi-sig bridge validation must be processed immediately
            MessagePriority::Critical
        } else if topic.contains("/compute-power") {
            // v9.1.0: Compute power announcements — Low priority (informational)
            MessagePriority::Low
        } else if topic.contains("/mining-solutions") {
            MessagePriority::Low
        } else {
            MessagePriority::Lowest
        }
    }
}

/// Queued message with priority and metadata
#[derive(Debug)]
pub struct QueuedMessage {
    pub topic: String,
    pub data: Vec<u8>,
    pub priority: MessagePriority,
    pub enqueued_at: Instant,
    pub retry_count: u8,
}

impl Eq for QueuedMessage {}

impl PartialEq for QueuedMessage {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.enqueued_at == other.enqueued_at
    }
}

impl Ord for QueuedMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then older messages first
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.enqueued_at.cmp(&self.enqueued_at),
            ordering => ordering,
        }
    }
}

impl PartialOrd for QueuedMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Rate limiter for message types
#[derive(Debug)]
pub struct RateLimiter {
    /// Minimum interval between messages of this type (milliseconds)
    min_interval_ms: u64,
    /// Last message timestamp (milliseconds since epoch)
    last_message_ms: AtomicU64,
    /// Messages dropped due to rate limiting
    dropped_count: AtomicU64,
}

impl RateLimiter {
    pub fn new(min_interval_ms: u64) -> Self {
        Self {
            min_interval_ms,
            last_message_ms: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
        }
    }

    /// Check if a message can be sent (returns true if allowed)
    pub fn try_acquire(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let last_ms = self.last_message_ms.load(Ordering::Relaxed);
        let elapsed = now_ms.saturating_sub(last_ms);

        if elapsed >= self.min_interval_ms {
            self.last_message_ms.store(now_ms, Ordering::Relaxed);
            true
        } else {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }
}

/// Queue statistics
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub current_size: usize,
    pub max_size: usize,
    pub messages_enqueued: u64,
    pub messages_sent: u64,
    pub messages_dropped: u64,
    pub messages_expired: u64,
    pub rate_limited: u64,
    pub critical_queue_size: usize,
    pub high_queue_size: usize,
    pub normal_queue_size: usize,
    pub low_queue_size: usize,
}

/// Configuration for the gossipsub queue
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum total queue size
    pub max_queue_size: usize,
    /// Maximum age of messages before expiry (seconds)
    pub max_message_age_secs: u64,
    /// Rate limits by priority (milliseconds between messages)
    pub rate_limit_critical_ms: u64,
    pub rate_limit_high_ms: u64,
    pub rate_limit_normal_ms: u64,
    pub rate_limit_low_ms: u64,
    pub rate_limit_lowest_ms: u64,
    /// Target drain rate per second
    pub target_drain_rate: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10_000,
            max_message_age_secs: 30,
            // Rate limits: blocks have no limit, mining solutions are heavily limited
            rate_limit_critical_ms: 0,    // No rate limit for critical (blocks)
            rate_limit_high_ms: 10,       // 100/sec max for peer heights
            rate_limit_normal_ms: 50,     // 20/sec max for balance updates
            rate_limit_low_ms: 100,       // 10/sec max for mining solutions
            rate_limit_lowest_ms: 500,    // 2/sec max for telemetry
            target_drain_rate: 1000,      // Process up to 1000 messages/sec
        }
    }
}

/// Sophisticated gossipsub message queue
pub struct GossipsubQueue {
    /// Priority queue for messages
    queue: Mutex<BinaryHeap<QueuedMessage>>,
    /// Rate limiters by priority
    rate_limiters: [RateLimiter; 5],
    /// Configuration
    config: QueueConfig,
    /// Statistics
    messages_enqueued: AtomicU64,
    messages_sent: AtomicU64,
    messages_dropped: AtomicU64,
    messages_expired: AtomicU64,
    current_size: AtomicUsize,
}

impl GossipsubQueue {
    /// Create a new gossipsub queue with default configuration
    pub fn new() -> Self {
        Self::with_config(QueueConfig::default())
    }

    /// Create a new gossipsub queue with custom configuration
    pub fn with_config(config: QueueConfig) -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::new()),
            rate_limiters: [
                RateLimiter::new(config.rate_limit_critical_ms),  // Critical
                RateLimiter::new(config.rate_limit_high_ms),      // High
                RateLimiter::new(config.rate_limit_normal_ms),    // Normal
                RateLimiter::new(config.rate_limit_low_ms),       // Low
                RateLimiter::new(config.rate_limit_lowest_ms),    // Lowest
            ],
            config,
            messages_enqueued: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_dropped: AtomicU64::new(0),
            messages_expired: AtomicU64::new(0),
            current_size: AtomicUsize::new(0),
        }
    }

    /// Get the rate limiter index for a priority
    fn rate_limiter_index(priority: MessagePriority) -> usize {
        match priority {
            MessagePriority::Critical => 0,
            MessagePriority::High => 1,
            MessagePriority::Normal => 2,
            MessagePriority::Low => 3,
            MessagePriority::Lowest => 4,
        }
    }

    /// Enqueue a message for sending
    /// Returns Ok(()) if enqueued, Err(reason) if dropped
    pub fn enqueue(&self, topic: String, data: Vec<u8>) -> Result<(), String> {
        let priority = MessagePriority::from_topic(&topic);

        // Check rate limit
        let limiter_idx = Self::rate_limiter_index(priority);
        if !self.rate_limiters[limiter_idx].try_acquire() {
            trace!(
                "Message rate limited: topic={}, priority={:?}",
                topic, priority
            );
            return Err("Rate limited".to_string());
        }

        let current_size = self.current_size.load(Ordering::Relaxed);

        // Queue overflow protection
        if current_size >= self.config.max_queue_size {
            // Drop lowest priority messages first
            if priority == MessagePriority::Lowest || priority == MessagePriority::Low {
                self.messages_dropped.fetch_add(1, Ordering::Relaxed);
                return Err("Queue full, low priority dropped".to_string());
            }

            // Try to make room by expiring old messages
            self.expire_old_messages();

            if self.current_size.load(Ordering::Relaxed) >= self.config.max_queue_size {
                self.messages_dropped.fetch_add(1, Ordering::Relaxed);
                return Err("Queue full after cleanup".to_string());
            }
        }

        let message = QueuedMessage {
            topic,
            data,
            priority,
            enqueued_at: Instant::now(),
            retry_count: 0,
        };

        {
            let mut queue = self.queue.lock().unwrap();
            queue.push(message);
        }

        self.current_size.fetch_add(1, Ordering::Relaxed);
        self.messages_enqueued.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Drain up to `max_count` messages from the queue (highest priority first)
    /// Skips expired messages automatically.
    pub fn drain_batch(&self, max_count: usize) -> Vec<QueuedMessage> {
        let mut batch = Vec::with_capacity(max_count);
        let mut queue = self.queue.lock().unwrap();
        let max_age = Duration::from_secs(self.config.max_message_age_secs);

        while batch.len() < max_count {
            match queue.pop() {
                Some(msg) => {
                    self.current_size.fetch_sub(1, Ordering::Relaxed);
                    if msg.enqueued_at.elapsed() < max_age {
                        self.messages_sent.fetch_add(1, Ordering::Relaxed);
                        batch.push(msg);
                    } else {
                        self.messages_expired.fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => break,
            }
        }

        batch
    }

    /// Get the next message to send (highest priority, oldest first)
    pub fn dequeue(&self) -> Option<QueuedMessage> {
        let mut queue = self.queue.lock().unwrap();

        while let Some(msg) = queue.pop() {
            self.current_size.fetch_sub(1, Ordering::Relaxed);

            // Check if message has expired
            if msg.enqueued_at.elapsed().as_secs() > self.config.max_message_age_secs {
                self.messages_expired.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            self.messages_sent.fetch_add(1, Ordering::Relaxed);
            return Some(msg);
        }

        None
    }

    /// Expire old messages from the queue
    fn expire_old_messages(&self) {
        let mut queue = self.queue.lock().unwrap();
        let max_age = Duration::from_secs(self.config.max_message_age_secs);

        let mut new_queue = BinaryHeap::new();
        let mut expired_count = 0u64;

        while let Some(msg) = queue.pop() {
            if msg.enqueued_at.elapsed() < max_age {
                new_queue.push(msg);
            } else {
                expired_count += 1;
            }
        }

        *queue = new_queue;
        self.current_size.store(queue.len(), Ordering::Relaxed);
        self.messages_expired.fetch_add(expired_count, Ordering::Relaxed);

        if expired_count > 0 {
            debug!("Expired {} old messages from gossipsub queue", expired_count);
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let queue = self.queue.lock().unwrap();

        let mut critical_count = 0;
        let mut high_count = 0;
        let mut normal_count = 0;
        let mut low_count = 0;

        for msg in queue.iter() {
            match msg.priority {
                MessagePriority::Critical => critical_count += 1,
                MessagePriority::High => high_count += 1,
                MessagePriority::Normal => normal_count += 1,
                MessagePriority::Low | MessagePriority::Lowest => low_count += 1,
            }
        }

        let rate_limited: u64 = self.rate_limiters.iter().map(|r| r.dropped_count()).sum();

        QueueStats {
            current_size: queue.len(),
            max_size: self.config.max_queue_size,
            messages_enqueued: self.messages_enqueued.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_dropped: self.messages_dropped.load(Ordering::Relaxed),
            messages_expired: self.messages_expired.load(Ordering::Relaxed),
            rate_limited,
            critical_queue_size: critical_count,
            high_queue_size: high_count,
            normal_queue_size: normal_count,
            low_queue_size: low_count,
        }
    }

    /// Check if backpressure should be applied (queue getting full)
    pub fn should_backpressure(&self) -> bool {
        let current = self.current_size.load(Ordering::Relaxed);
        current >= self.config.max_queue_size * 80 / 100 // 80% threshold
    }

    /// Get the current queue fill percentage
    pub fn fill_percentage(&self) -> f64 {
        let current = self.current_size.load(Ordering::Relaxed) as f64;
        let max = self.config.max_queue_size as f64;
        (current / max) * 100.0
    }
}

impl Default for GossipsubQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Network throttle mode controlled by Q_NETWORK_THROTTLE env var.
///
/// - `full` (default): No rate limiting on any message type. Maximum throughput.
/// - `conservative`: Rate-limited mode for bandwidth-constrained nodes.
///
/// Example: Q_NETWORK_THROTTLE=conservative ./q-api-server
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThrottleMode {
    /// No rate limiting — maximum network throughput (default)
    Full,
    /// Conservative rate limiting for bandwidth-constrained nodes
    Conservative,
}

impl ThrottleMode {
    pub fn from_env() -> Self {
        match std::env::var("Q_NETWORK_THROTTLE")
            .unwrap_or_else(|_| "full".to_string())
            .to_lowercase()
            .as_str()
        {
            "conservative" | "slow" | "limited" => ThrottleMode::Conservative,
            _ => ThrottleMode::Full, // "full", "fast", "unlimited", or anything else
        }
    }
}

/// Global gossipsub queue instance
lazy_static! {
    static ref GOSSIPSUB_QUEUE: GossipsubQueue = {
        let throttle = ThrottleMode::from_env();
        let config = match throttle {
            ThrottleMode::Full => {
                // Full throttle: minimal rate limiting — only cap high-priority (peer-height)
                // v1.0.2-safe: rate_limit_high_ms=5 (200/sec max) prevents O(N²) forwarding
                // amplification with 60+ peers each announcing heights every 5s
                QueueConfig {
                    max_queue_size: std::env::var("Q_GOSSIPSUB_QUEUE_SIZE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(50_000),
                    max_message_age_secs: 60,
                    rate_limit_critical_ms: 0,
                    rate_limit_high_ms: 5,
                    rate_limit_normal_ms: 0,
                    rate_limit_low_ms: 0,
                    rate_limit_lowest_ms: 0,
                    target_drain_rate: 10_000,
                }
            }
            ThrottleMode::Conservative => {
                // Conservative: rate-limited for bandwidth-constrained nodes
                QueueConfig {
                    max_queue_size: std::env::var("Q_GOSSIPSUB_QUEUE_SIZE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(10_000),
                    max_message_age_secs: 30,
                    rate_limit_critical_ms: 0,    // Blocks always unlimited
                    rate_limit_high_ms: 10,       // 100/sec for peer heights
                    rate_limit_normal_ms: 50,     // 20/sec for balance updates
                    rate_limit_low_ms: std::env::var("Q_MINING_SOLUTION_RATE_LIMIT_MS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(100),          // 10/sec for mining solutions
                    rate_limit_lowest_ms: 500,    // 2/sec for telemetry
                    target_drain_rate: 1000,
                }
            }
        };
        info!(
            "🌐 Gossipsub queue: mode={:?}, max_size={}, rate_limits=[{},{},{},{},{}]ms",
            throttle, config.max_queue_size,
            config.rate_limit_critical_ms, config.rate_limit_high_ms,
            config.rate_limit_normal_ms, config.rate_limit_low_ms, config.rate_limit_lowest_ms
        );
        GossipsubQueue::with_config(config)
    };
}

/// Get the global gossipsub queue
pub fn gossipsub_queue() -> &'static GossipsubQueue {
    &*GOSSIPSUB_QUEUE
}

/// Quick function to check if a message should be rate limited
pub fn should_rate_limit_message(topic: &str) -> bool {
    let priority = MessagePriority::from_topic(topic);
    let limiter_idx = GossipsubQueue::rate_limiter_index(priority);
    !GOSSIPSUB_QUEUE.rate_limiters[limiter_idx].try_acquire()
}

/// Log queue statistics periodically
pub fn log_queue_stats() {
    let stats = GOSSIPSUB_QUEUE.stats();
    if stats.current_size > 0 || stats.rate_limited > 0 || stats.messages_dropped > 0 {
        info!(
            "Gossipsub queue: size={}/{} ({:.1}%), sent={}, dropped={}, rate_limited={}, expired={}",
            stats.current_size,
            stats.max_size,
            GOSSIPSUB_QUEUE.fill_percentage(),
            stats.messages_sent,
            stats.messages_dropped,
            stats.rate_limited,
            stats.messages_expired
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
        assert!(MessagePriority::Low > MessagePriority::Lowest);
    }

    #[test]
    fn test_priority_from_topic() {
        assert_eq!(MessagePriority::from_topic("/qnk/testnet/blocks"), MessagePriority::Critical);
        assert_eq!(MessagePriority::from_topic("/qnk/testnet/peer-heights"), MessagePriority::High);
        assert_eq!(MessagePriority::from_topic("/qnk/testnet/balance"), MessagePriority::Normal);
        assert_eq!(MessagePriority::from_topic("/qnk/testnet/mining-solutions"), MessagePriority::Low);
        assert_eq!(MessagePriority::from_topic("/qnk/testnet/other"), MessagePriority::Lowest);
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let queue = GossipsubQueue::with_config(QueueConfig {
            rate_limit_critical_ms: 0,
            rate_limit_high_ms: 0,
            rate_limit_normal_ms: 0,
            rate_limit_low_ms: 0,
            rate_limit_lowest_ms: 0,
            ..QueueConfig::default()
        });

        // Enqueue messages with different priorities
        queue.enqueue("/blocks".to_string(), vec![1, 2, 3]).unwrap();
        queue.enqueue("/mining-solutions".to_string(), vec![4, 5, 6]).unwrap();
        queue.enqueue("/peer-heights".to_string(), vec![7, 8, 9]).unwrap();

        // Dequeue should return highest priority first
        let msg1 = queue.dequeue().unwrap();
        assert!(msg1.topic.contains("blocks"));

        let msg2 = queue.dequeue().unwrap();
        assert!(msg2.topic.contains("peer-heights"));

        let msg3 = queue.dequeue().unwrap();
        assert!(msg3.topic.contains("mining-solutions"));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(100); // 100ms interval

        // First call should succeed
        assert!(limiter.try_acquire());

        // Immediate second call should fail
        assert!(!limiter.try_acquire());

        // After waiting, should succeed again
        std::thread::sleep(std::time::Duration::from_millis(110));
        assert!(limiter.try_acquire());
    }

    #[test]
    fn test_queue_overflow_protection() {
        let queue = GossipsubQueue::with_config(QueueConfig {
            max_queue_size: 10,
            rate_limit_critical_ms: 0,
            rate_limit_high_ms: 0,
            rate_limit_normal_ms: 0,
            rate_limit_low_ms: 0,
            rate_limit_lowest_ms: 0,
            ..QueueConfig::default()
        });

        // Fill queue
        for i in 0..10 {
            queue.enqueue("/blocks".to_string(), vec![i as u8]).unwrap();
        }

        // Low priority should be dropped when queue is full
        let result = queue.enqueue("/mining-solutions".to_string(), vec![100]);
        assert!(result.is_err());

        // High priority might still get through after expiring old messages
        // (in this case, no messages are expired since they're all fresh)
    }
}
