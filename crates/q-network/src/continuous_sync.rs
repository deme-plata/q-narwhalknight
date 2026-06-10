/// 🚀 Project APOLLO Phase 4: SLINGSHOT - Continuous Stream Protocol (SCRAMJET FLOW)
///
/// Replace request-response sync with continuous streaming:
/// - Upstream: Signal demand continuously (block ranges we need)
/// - Downstream: Receive blocks continuously (no wait for responses)
/// - No request-response overhead (RTT savings)
///
/// Like a scramjet engine: continuous flow, no stop-start cycles.
///
/// Traditional sync: Request → Wait RTT → Response → Process → Request → Wait RTT...
/// Scramjet sync:    Stream demand ──────────→
///                   ←────────── Continuous block flow
///
/// Expected improvement: 30-50% reduction in sync time by eliminating RTT waits

use anyhow::{Context, Result, bail};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Demand signal sent upstream to peers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemandSignal {
    /// Range of blocks we need
    pub range: Range<u64>,

    /// Priority (higher = more urgent)
    pub priority: u8,

    /// Request ID for correlation
    pub request_id: u64,

    /// Maximum blocks to send per message
    pub max_batch_size: usize,

    /// Timestamp for latency measurement
    pub timestamp: u64,
}

impl DemandSignal {
    pub fn new(range: Range<u64>, priority: u8) -> Self {
        Self {
            range,
            priority,
            request_id: rand::random(),
            max_batch_size: 500,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

/// Block data flowing downstream
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockFlow {
    /// Height of this block
    pub height: u64,

    /// Serialized block data (compressed)
    pub data: Vec<u8>,

    /// Correlation to demand signal
    pub request_id: u64,

    /// Sender peer ID
    pub from_peer: String,

    /// Indicates end of stream for this range
    pub is_final: bool,
}

/// Flow control for continuous streaming
#[derive(Clone, Debug)]
pub struct FlowControl {
    /// Maximum outstanding demands per peer
    pub max_outstanding_per_peer: usize,

    /// Maximum total outstanding demands
    pub max_total_outstanding: usize,

    /// Demand batch size (blocks per demand signal)
    pub demand_batch_size: u64,

    /// Receive buffer size (blocks to buffer before backpressure)
    pub receive_buffer_size: usize,

    /// Timeout for demand fulfillment
    pub demand_timeout: Duration,

    /// Enable adaptive flow control
    pub adaptive: bool,
}

impl Default for FlowControl {
    fn default() -> Self {
        Self {
            max_outstanding_per_peer: 8,
            max_total_outstanding: 64,
            demand_batch_size: 500,
            receive_buffer_size: 10_000,
            demand_timeout: Duration::from_secs(30),
            adaptive: true,
        }
    }
}

/// Outstanding demand tracking
#[derive(Clone, Debug)]
struct OutstandingDemand {
    signal: DemandSignal,
    sent_at: Instant,
    peer_id: String,
    blocks_received: u64,
}

/// Continuous sync engine (SCRAMJET FLOW)
pub struct ScramjetSync {
    /// Our node ID
    node_id: String,

    /// Flow control settings
    flow_control: FlowControl,

    /// Demand signal sender (to network layer)
    demand_tx: mpsc::Sender<(String, DemandSignal)>,

    /// Block receiver (from network layer)
    block_rx: mpsc::Receiver<BlockFlow>,

    /// Outstanding demands
    outstanding: Arc<RwLock<HashMap<u64, OutstandingDemand>>>,

    /// Per-peer outstanding count
    peer_outstanding: Arc<RwLock<HashMap<String, usize>>>,

    /// Received blocks buffer (for reordering)
    receive_buffer: Arc<RwLock<VecDeque<BlockFlow>>>,

    /// Current demand height (what we've requested up to)
    demand_height: Arc<RwLock<u64>>,

    /// Completed height (what we've processed)
    completed_height: Arc<RwLock<u64>>,

    /// Semaphore for total outstanding limit
    total_semaphore: Arc<Semaphore>,

    /// Metrics
    metrics: Arc<RwLock<ScramjetMetrics>>,
}

/// Scramjet sync metrics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScramjetMetrics {
    /// Total demands sent
    pub demands_sent: u64,

    /// Total blocks received
    pub blocks_received: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Demands timed out
    pub demands_timeout: u64,

    /// Current outstanding demands
    pub outstanding_count: usize,

    /// Average blocks per second
    pub blocks_per_second: f64,

    /// Average latency (ms)
    pub avg_latency_ms: f64,

    /// Buffer utilization (0.0 - 1.0)
    pub buffer_utilization: f64,
}

impl ScramjetSync {
    /// Create new scramjet sync engine
    pub fn new(
        node_id: String,
        flow_control: FlowControl,
        demand_tx: mpsc::Sender<(String, DemandSignal)>,
        block_rx: mpsc::Receiver<BlockFlow>,
    ) -> Self {
        let max_outstanding = flow_control.max_total_outstanding;
        Self {
            node_id,
            flow_control,
            demand_tx,
            block_rx,
            outstanding: Arc::new(RwLock::new(HashMap::new())),
            peer_outstanding: Arc::new(RwLock::new(HashMap::new())),
            receive_buffer: Arc::new(RwLock::new(VecDeque::new())),
            demand_height: Arc::new(RwLock::new(0)),
            completed_height: Arc::new(RwLock::new(0)),
            total_semaphore: Arc::new(Semaphore::new(max_outstanding)),
            metrics: Arc::new(RwLock::new(ScramjetMetrics::default())),
        }
    }

    /// Start continuous sync from current height to target
    pub async fn sync_range(
        &self,
        start_height: u64,
        target_height: u64,
        peers: Vec<String>,
    ) -> Result<()> {
        if peers.is_empty() {
            bail!("No peers available for sync");
        }

        info!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );
        info!(
            "🚀 [SCRAMJET FLOW] Starting continuous sync: {} → {}",
            start_height, target_height
        );
        info!(
            "   Peers: {}, Outstanding limit: {}, Batch size: {}",
            peers.len(),
            self.flow_control.max_total_outstanding,
            self.flow_control.demand_batch_size
        );
        info!(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        );

        // Initialize heights
        *self.demand_height.write().await = start_height;
        *self.completed_height.write().await = start_height;

        let sync_start = Instant::now();

        // Spawn demand generator
        let demand_handle = self.spawn_demand_generator(target_height, peers);

        // Spawn block processor
        let processor_handle = self.spawn_block_processor();

        // Spawn timeout checker
        let timeout_handle = self.spawn_timeout_checker();

        // Wait for completion or error
        let completed = self.completed_height.clone();
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;

            let current = *completed.read().await;
            if current >= target_height {
                info!(
                    "✅ [SCRAMJET FLOW] Sync complete! {} → {} in {:?}",
                    start_height,
                    target_height,
                    sync_start.elapsed()
                );
                break;
            }

            // Log progress periodically
            if sync_start.elapsed().as_secs() % 10 == 0 {
                let metrics = self.metrics.read().await;
                info!(
                    "📊 [SCRAMJET] Progress: {} / {} ({:.1}%), {:.1} blocks/sec",
                    current,
                    target_height,
                    (current - start_height) as f64 / (target_height - start_height) as f64 * 100.0,
                    metrics.blocks_per_second
                );
            }
        }

        Ok(())
    }

    /// Spawn demand generator task
    fn spawn_demand_generator(
        &self,
        target_height: u64,
        peers: Vec<String>,
    ) -> tokio::task::JoinHandle<()> {
        let demand_tx = self.demand_tx.clone();
        let demand_height = self.demand_height.clone();
        let outstanding = self.outstanding.clone();
        let peer_outstanding = self.peer_outstanding.clone();
        let total_semaphore = self.total_semaphore.clone();
        let metrics = self.metrics.clone();
        let flow_control = self.flow_control.clone();

        tokio::spawn(async move {
            let mut peer_idx = 0;

            loop {
                // Check if we've demanded up to target
                let current_demand = *demand_height.read().await;
                if current_demand >= target_height {
                    debug!("[SCRAMJET] All demands sent up to target {}", target_height);
                    break;
                }

                // Wait for available slot
                let permit = total_semaphore.acquire().await.unwrap();

                // Select peer (round-robin for simplicity, can be enhanced with momentum)
                let peer = peers[peer_idx % peers.len()].clone();
                peer_idx += 1;

                // Check per-peer limit
                {
                    let peer_count = peer_outstanding.read().await;
                    if peer_count.get(&peer).copied().unwrap_or(0)
                        >= flow_control.max_outstanding_per_peer
                    {
                        drop(permit);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                }

                // Generate demand signal
                let batch_size = flow_control.demand_batch_size;
                let mut current = demand_height.write().await;
                let start = *current;
                let end = (start + batch_size).min(target_height);
                *current = end;
                drop(current);

                let signal = DemandSignal::new(start..end, 5);
                let request_id = signal.request_id;

                // Track outstanding
                {
                    let mut os = outstanding.write().await;
                    os.insert(
                        request_id,
                        OutstandingDemand {
                            signal: signal.clone(),
                            sent_at: Instant::now(),
                            peer_id: peer.clone(),
                            blocks_received: 0,
                        },
                    );
                }

                {
                    let mut po = peer_outstanding.write().await;
                    *po.entry(peer.clone()).or_insert(0) += 1;
                }

                // Send demand
                if let Err(e) = demand_tx.send((peer.clone(), signal)).await {
                    error!("[SCRAMJET] Failed to send demand: {}", e);
                    break;
                }

                {
                    let mut m = metrics.write().await;
                    m.demands_sent += 1;
                    m.outstanding_count = outstanding.read().await.len();
                }

                // Don't drop permit - it will be returned when blocks are received
                std::mem::forget(permit);
            }
        })
    }

    /// Spawn block processor task
    fn spawn_block_processor(&self) -> tokio::task::JoinHandle<()> {
        // Note: This is a simplified version. In production, you'd process
        // blocks through the full storage pipeline.

        let outstanding = self.outstanding.clone();
        let peer_outstanding = self.peer_outstanding.clone();
        let completed_height = self.completed_height.clone();
        let metrics = self.metrics.clone();
        let total_semaphore = self.total_semaphore.clone();

        // Create a new receiver by cloning the arc pattern
        // In real impl, this would be handled differently
        let block_rx = self.receive_buffer.clone();

        tokio::spawn(async move {
            loop {
                // In real impl, we'd receive from block_rx channel
                // For now, simulate processing from buffer
                let block = {
                    let mut buf = block_rx.write().await;
                    buf.pop_front()
                };

                if let Some(block) = block {
                    // Update metrics
                    {
                        let mut m = metrics.write().await;
                        m.blocks_received += 1;
                        m.bytes_received += block.data.len() as u64;
                    }

                    // Update completed height
                    {
                        let mut completed = completed_height.write().await;
                        if block.height > *completed {
                            *completed = block.height;
                        }
                    }

                    // Handle request completion
                    if block.is_final {
                        if let Some(demand) = outstanding.write().await.remove(&block.request_id) {
                            // Update peer outstanding
                            {
                                let mut po = peer_outstanding.write().await;
                                if let Some(count) = po.get_mut(&demand.peer_id) {
                                    *count = count.saturating_sub(1);
                                }
                            }

                            // Return permit
                            total_semaphore.add_permits(1);

                            // Update latency metrics
                            let latency = demand.sent_at.elapsed().as_millis() as f64;
                            let mut m = metrics.write().await;
                            m.avg_latency_ms = 0.9 * m.avg_latency_ms + 0.1 * latency;
                        }
                    }
                } else {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        })
    }

    /// Spawn timeout checker task
    fn spawn_timeout_checker(&self) -> tokio::task::JoinHandle<()> {
        let outstanding = self.outstanding.clone();
        let peer_outstanding = self.peer_outstanding.clone();
        let metrics = self.metrics.clone();
        let total_semaphore = self.total_semaphore.clone();
        let timeout = self.flow_control.demand_timeout;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;

                let now = Instant::now();
                let mut to_remove = Vec::new();

                {
                    let os = outstanding.read().await;
                    for (request_id, demand) in os.iter() {
                        if now.duration_since(demand.sent_at) > timeout {
                            to_remove.push(*request_id);
                        }
                    }
                }

                if !to_remove.is_empty() {
                    let mut os = outstanding.write().await;
                    for request_id in to_remove {
                        if let Some(demand) = os.remove(&request_id) {
                            warn!(
                                "⚠️ [SCRAMJET] Demand {} timed out for range {:?}",
                                request_id, demand.signal.range
                            );

                            // Update peer outstanding
                            {
                                let mut po = peer_outstanding.write().await;
                                if let Some(count) = po.get_mut(&demand.peer_id) {
                                    *count = count.saturating_sub(1);
                                }
                            }

                            // Return permit
                            total_semaphore.add_permits(1);

                            // Update metrics
                            {
                                let mut m = metrics.write().await;
                                m.demands_timeout += 1;
                            }
                        }
                    }
                }
            }
        })
    }

    /// Handle incoming block from peer
    pub async fn handle_block(&self, block: BlockFlow) {
        // Add to receive buffer
        self.receive_buffer.write().await.push_back(block);

        // Update buffer utilization metric
        let buffer_len = self.receive_buffer.read().await.len();
        let mut m = self.metrics.write().await;
        m.buffer_utilization = buffer_len as f64 / self.flow_control.receive_buffer_size as f64;
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ScramjetMetrics {
        self.metrics.read().await.clone()
    }

    /// Get outstanding demand count
    pub async fn outstanding_count(&self) -> usize {
        self.outstanding.read().await.len()
    }

    /// Check if sync is active
    pub async fn is_active(&self) -> bool {
        !self.outstanding.read().await.is_empty()
    }
}

/// Server-side continuous flow handler
pub struct ScramjetServer {
    /// Our node ID
    node_id: String,

    /// Active streams per peer
    active_streams: Arc<RwLock<HashMap<String, ActiveStream>>>,

    /// Block sender (to network layer)
    block_tx: mpsc::Sender<(String, BlockFlow)>,
}

/// Active stream state for a peer
struct ActiveStream {
    /// Current demand being fulfilled
    current_demand: Option<DemandSignal>,

    /// Next height to send
    next_height: u64,

    /// Blocks sent for current demand
    blocks_sent: u64,
}

impl ScramjetServer {
    pub fn new(node_id: String, block_tx: mpsc::Sender<(String, BlockFlow)>) -> Self {
        Self {
            node_id,
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            block_tx,
        }
    }

    /// Handle demand signal from peer
    pub async fn handle_demand(
        &self,
        peer_id: String,
        demand: DemandSignal,
        block_provider: impl BlockProvider,
    ) -> Result<()> {
        debug!(
            "📥 [SCRAMJET SERVER] Demand from {}: {:?}",
            peer_id, demand.range
        );

        // Update active stream
        {
            let mut streams = self.active_streams.write().await;
            streams.insert(
                peer_id.clone(),
                ActiveStream {
                    current_demand: Some(demand.clone()),
                    next_height: demand.range.start,
                    blocks_sent: 0,
                },
            );
        }

        // Stream blocks continuously
        let mut height = demand.range.start;
        let end = demand.range.end;

        while height < end {
            let batch_end = (height + demand.max_batch_size as u64).min(end);

            // Fetch blocks
            let blocks = block_provider
                .get_blocks(height, batch_end)
                .await
                .context("Failed to fetch blocks for stream")?;

            // Send each block
            for (h, data) in blocks {
                let is_final = h + 1 >= end;
                let block_flow = BlockFlow {
                    height: h,
                    data,
                    request_id: demand.request_id,
                    from_peer: self.node_id.clone(),
                    is_final,
                };

                self.block_tx
                    .send((peer_id.clone(), block_flow))
                    .await
                    .context("Failed to send block to peer")?;
            }

            height = batch_end;
        }

        // Clean up stream
        self.active_streams.write().await.remove(&peer_id);

        Ok(())
    }

    /// Get active stream count
    pub async fn active_stream_count(&self) -> usize {
        self.active_streams.read().await.len()
    }
}

/// Block provider trait for server-side
#[async_trait::async_trait]
pub trait BlockProvider: Send + Sync {
    /// Get blocks in range (returns Vec<(height, serialized_data)>)
    async fn get_blocks(&self, start: u64, end: u64) -> Result<Vec<(u64, Vec<u8>)>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demand_signal_creation() {
        let signal = DemandSignal::new(100..200, 5);
        assert_eq!(signal.range, 100..200);
        assert_eq!(signal.priority, 5);
        assert!(signal.request_id != 0);
    }

    #[test]
    fn test_flow_control_defaults() {
        let fc = FlowControl::default();
        assert_eq!(fc.max_outstanding_per_peer, 8);
        assert_eq!(fc.max_total_outstanding, 64);
        assert_eq!(fc.demand_batch_size, 500);
    }

    #[tokio::test]
    async fn test_scramjet_metrics() {
        let metrics = ScramjetMetrics::default();
        assert_eq!(metrics.demands_sent, 0);
        assert_eq!(metrics.blocks_received, 0);
    }
}
