/// Request Pipelining for Turbo Sync
///
/// Phase 2: Network Optimization - Request Pipelining (Week 2-3)
/// Target: 12.8 → 19.2 blocks/s (+50% improvement)
///
/// Architecture:
/// - Pipeline depth: 2 (conservative start, prevents peer overload)
/// - Adaptive window sizing based on RTT measurements
/// - Flow control to prevent overwhelming slow peers
/// - Request coalescing to reduce protocol overhead
///
/// Key Innovation:
/// Instead of sequential request/response cycles:
///   Request 1 → Response 1 → Request 2 → Response 2  (slow)
///
/// We pipeline multiple requests:
///   Request 1 → Request 2 → Response 1 → Response 2  (fast)
///
/// This hides network latency and maximizes throughput.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, warn};

use q_types::block::QBlock;

/// Configuration for request pipelining
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Initial pipeline depth (number of concurrent in-flight requests)
    /// Conservative start: 2 (can adapt up to 8 based on RTT)
    pub initial_depth: usize,

    /// Minimum pipeline depth (never go below this)
    pub min_depth: usize,

    /// Maximum pipeline depth (never exceed this, prevents peer overload)
    pub max_depth: usize,

    /// Target RTT for adaptive window sizing (milliseconds)
    /// If RTT < target, increase depth. If RTT > target, decrease depth.
    pub target_rtt_ms: u64,

    /// RTT measurement window (how many samples to average)
    pub rtt_window_size: usize,

    /// Timeout for individual requests
    pub request_timeout: Duration,

    /// Enable flow control (backoff when peer is slow)
    pub enable_flow_control: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        // v1.4.12-beta: AGGRESSIVE PIPELINING for 100+ blocks/s sync
        // Testing showed 2-depth was limiting throughput on high-bandwidth links
        // New values maximize parallelism while still preventing peer overload
        Self {
            initial_depth: 4,        // v1.4.12: Start with 4 concurrent requests (was 2)
            min_depth: 2,            // v1.4.12: Never go below 2 (was 1)
            max_depth: 16,           // v1.4.12: Allow up to 16 parallel (was 8)
            target_rtt_ms: 50,       // v1.4.12: Target 50ms RTT for faster adaptation (was 100)
            rtt_window_size: 10,     // Average last 10 RTT samples
            request_timeout: Duration::from_secs(30),
            enable_flow_control: true,
        }
    }
}

/// RTT (Round Trip Time) tracker for adaptive window sizing
#[derive(Debug)]
struct RttTracker {
    samples: VecDeque<Duration>,
    max_samples: usize,
}

impl RttTracker {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record a new RTT sample
    fn record(&mut self, rtt: Duration) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(rtt);
    }

    /// Get average RTT
    fn avg_rtt(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }

        let total: Duration = self.samples.iter().sum();
        Some(total / self.samples.len() as u32)
    }

    /// Get latest RTT
    fn latest_rtt(&self) -> Option<Duration> {
        self.samples.back().copied()
    }
}

/// Request state tracking
#[derive(Debug)]
struct InFlightRequest {
    start_height: u64,
    end_height: u64,
    sent_at: Instant,
}

/// Pipeline Manager - coordinates pipelined requests
pub struct PipelineManager {
    config: PipelineConfig,
    current_depth: Arc<Mutex<usize>>,
    rtt_tracker: Arc<Mutex<RttTracker>>,
    in_flight: Arc<Mutex<Vec<InFlightRequest>>>,
    semaphore: Arc<Semaphore>,
}

impl PipelineManager {
    pub fn new(config: PipelineConfig) -> Self {
        let initial_depth = config.initial_depth;

        Self {
            config,
            current_depth: Arc::new(Mutex::new(initial_depth)),
            rtt_tracker: Arc::new(Mutex::new(RttTracker::new(10))),
            in_flight: Arc::new(Mutex::new(Vec::new())),
            semaphore: Arc::new(Semaphore::new(initial_depth)),
        }
    }

    /// Acquire permission to send a request (blocks if pipeline is full)
    pub async fn acquire_slot(&self) -> Result<PipelineSlot> {
        let permit = self.semaphore.clone().acquire_owned().await
            .context("Failed to acquire pipeline slot")?;

        Ok(PipelineSlot {
            permit,
            manager: Arc::new(self.clone_ref()),
        })
    }

    /// Record that a request was sent
    pub async fn record_request_sent(&self, start_height: u64, end_height: u64) {
        let mut in_flight = self.in_flight.lock().await;
        in_flight.push(InFlightRequest {
            start_height,
            end_height,
            sent_at: Instant::now(),
        });
    }

    /// Record that a response was received (updates RTT and adaptive window)
    pub async fn record_response_received(&self, start_height: u64) {
        let mut in_flight = self.in_flight.lock().await;

        // Find and remove the completed request
        if let Some(pos) = in_flight.iter().position(|r| r.start_height == start_height) {
            let request = in_flight.remove(pos);
            let rtt = request.sent_at.elapsed();

            // Update RTT tracker
            let mut rtt_tracker = self.rtt_tracker.lock().await;
            rtt_tracker.record(rtt);

            // Adaptive window sizing (only if flow control enabled)
            if self.config.enable_flow_control {
                self.adjust_pipeline_depth(rtt).await;
            }

            debug!(
                "📊 [PIPELINE] Request completed in {:?} (depth: {})",
                rtt,
                *self.current_depth.lock().await
            );
        }
    }

    /// Adjust pipeline depth based on RTT
    async fn adjust_pipeline_depth(&self, latest_rtt: Duration) {
        let target_rtt = Duration::from_millis(self.config.target_rtt_ms);
        let mut depth = self.current_depth.lock().await;

        // If RTT is better than target, try increasing depth
        if latest_rtt < target_rtt && *depth < self.config.max_depth {
            *depth += 1;
            info!("⬆️  [PIPELINE] Increased depth to {} (RTT: {:?} < target {:?})",
                  *depth, latest_rtt, target_rtt);

            // Update semaphore permits
            self.semaphore.add_permits(1);
        }
        // If RTT is worse than target, decrease depth
        else if latest_rtt > target_rtt * 2 && *depth > self.config.min_depth {
            *depth -= 1;
            warn!("⬇️  [PIPELINE] Decreased depth to {} (RTT: {:?} > target {:?})",
                  *depth, latest_rtt, target_rtt);

            // Don't need to remove permits - they'll naturally be consumed
        }
    }

    /// Get current pipeline depth
    pub async fn current_depth(&self) -> usize {
        *self.current_depth.lock().await
    }

    /// Get average RTT
    pub async fn avg_rtt(&self) -> Option<Duration> {
        self.rtt_tracker.lock().await.avg_rtt()
    }

    /// Get number of in-flight requests
    pub async fn in_flight_count(&self) -> usize {
        self.in_flight.lock().await.len()
    }

    /// Clone reference (for use in PipelineSlot)
    fn clone_ref(&self) -> Self {
        Self {
            config: self.config.clone(),
            current_depth: Arc::clone(&self.current_depth),
            rtt_tracker: Arc::clone(&self.rtt_tracker),
            in_flight: Arc::clone(&self.in_flight),
            semaphore: Arc::clone(&self.semaphore),
        }
    }
}

/// Pipeline slot - represents permission to send a request
pub struct PipelineSlot {
    permit: tokio::sync::OwnedSemaphorePermit,
    manager: Arc<PipelineManager>,
}

impl PipelineSlot {
    /// Mark request as sent
    pub async fn sent(&self, start_height: u64, end_height: u64) {
        self.manager.record_request_sent(start_height, end_height).await;
    }

    /// Mark response as received
    pub async fn received(&self, start_height: u64) {
        self.manager.record_response_received(start_height).await;
    }
}

// Drop the permit when PipelineSlot goes out of scope
impl Drop for PipelineSlot {
    fn drop(&mut self) {
        // Permit is automatically returned to semaphore
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_slot_acquisition() {
        let config = PipelineConfig {
            initial_depth: 2,
            ..Default::default()
        };

        let manager = PipelineManager::new(config);

        // Should be able to acquire 2 slots (initial depth)
        let slot1 = manager.acquire_slot().await.unwrap();
        let slot2 = manager.acquire_slot().await.unwrap();

        // Should block on 3rd slot (would timeout)
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            manager.acquire_slot()
        ).await;

        assert!(result.is_err(), "Should timeout waiting for 3rd slot");

        // Drop slots to release permits
        drop(slot1);
        drop(slot2);
    }

    #[tokio::test]
    async fn test_rtt_tracking() {
        let config = PipelineConfig::default();
        let manager = PipelineManager::new(config);

        // Simulate request/response
        let slot = manager.acquire_slot().await.unwrap();
        slot.sent(1, 10).await;

        tokio::time::sleep(Duration::from_millis(50)).await;

        slot.received(1).await;

        // Check RTT was recorded
        let avg_rtt = manager.avg_rtt().await;
        assert!(avg_rtt.is_some());
        assert!(avg_rtt.unwrap() >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_adaptive_depth() {
        let config = PipelineConfig {
            initial_depth: 2,
            target_rtt_ms: 100,
            enable_flow_control: true,
            ..Default::default()
        };

        let manager = PipelineManager::new(config);

        // Simulate fast RTT (should increase depth)
        manager.record_response_received(1).await;
        let mut rtt_tracker = manager.rtt_tracker.lock().await;
        rtt_tracker.record(Duration::from_millis(50));
        drop(rtt_tracker);

        manager.adjust_pipeline_depth(Duration::from_millis(50)).await;

        let depth = manager.current_depth().await;
        assert_eq!(depth, 3, "Depth should increase with fast RTT");
    }
}
