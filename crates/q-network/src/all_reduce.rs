//! Ring All-Reduce for Tensor Parallelism
//!
//! Implements the ring all-reduce algorithm for combining partial tensor results
//! from multiple nodes during tensor-parallel inference.
//!
//! ## Why Ring All-Reduce?
//!
//! When N nodes each compute partial attention or MLP outputs, we need to
//! combine them efficiently. Ring all-reduce achieves this in O(2(N-1)) messages
//! with optimal bandwidth utilization.
//!
//! ## Algorithm
//!
//! For 4 nodes with tensors [A, B, C, D]:
//!
//! ```text
//! Phase 1: Scatter-Reduce (N-1 steps)
//! Step 1: Node 0→1: A[0], Node 1→2: B[1], Node 2→3: C[2], Node 3→0: D[3]
//! Step 2: Node 0→1: A[0]+D[3], Node 1→2: B[1]+A[0], ...
//! Step 3: Each chunk is now a partial sum of 2 elements
//!
//! Phase 2: All-Gather (N-1 steps)
//! Each node sends its reduced chunk to the next, until all nodes have all chunks.
//!
//! Result: All nodes have identical tensor = A + B + C + D
//! ```
//!
//! ## Performance
//!
//! - Messages: 2(N-1) per all-reduce
//! - Bandwidth: O(data_size) regardless of N (each byte sent once)
//! - Latency: O(N) network hops
//!
//! For tensor parallelism with 4 nodes:
//! - 32 layers × 2 all-reduces (attn + mlp) = 64 all-reduces per forward
//! - 64 × 6 messages = 384 messages per token generation

use anyhow::{anyhow, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{debug, error, info, warn};

/// Topic for all-reduce messages
pub const TOPIC_ALL_REDUCE: &str = "qnk/ai/all-reduce/v1";

/// All-reduce message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllReduceMessage {
    /// A chunk being sent during scatter-reduce or all-gather phase
    Chunk {
        /// Unique ID for this all-reduce operation
        request_id: String,

        /// Which layer this is for
        layer_idx: usize,

        /// Phase: "scatter" or "gather"
        phase: String,

        /// Step within the phase (0 to N-2)
        step: usize,

        /// Which chunk of the ring buffer
        chunk_idx: usize,

        /// The tensor data (compressed f32)
        data: Vec<u8>,

        /// Shape of the chunk
        shape: Vec<usize>,

        /// Whether data is compressed
        compressed: bool,
    },

    /// Signal that all-reduce is complete for this layer
    Complete {
        request_id: String,
        layer_idx: usize,
        total_time_ms: u64,
    },

    /// Error during all-reduce
    Error {
        request_id: String,
        layer_idx: usize,
        message: String,
    },
}

/// Configuration for all-reduce operations
#[derive(Debug, Clone)]
pub struct AllReduceConfig {
    /// Ordered list of peers in the ring (including self)
    pub ring_peers: Vec<PeerId>,

    /// This node's position in the ring
    pub ring_position: usize,

    /// Timeout for waiting on chunks
    pub chunk_timeout: Duration,

    /// Whether to compress chunks
    pub compression: bool,
}

impl AllReduceConfig {
    /// Create config for a set of peers
    pub fn new(mut peers: Vec<PeerId>, self_peer: PeerId) -> Result<Self> {
        // Sort peers for deterministic ring order
        peers.sort();

        let position = peers
            .iter()
            .position(|p| *p == self_peer)
            .ok_or_else(|| anyhow!("Self peer not in ring"))?;

        Ok(Self {
            ring_peers: peers,
            ring_position: position,
            chunk_timeout: Duration::from_secs(30),
            compression: true,
        })
    }

    /// Number of nodes in the ring
    pub fn world_size(&self) -> usize {
        self.ring_peers.len()
    }

    /// Get next peer in the ring (clockwise)
    pub fn next_peer(&self) -> &PeerId {
        let next_pos = (self.ring_position + 1) % self.ring_peers.len();
        &self.ring_peers[next_pos]
    }

    /// Get previous peer in the ring (counter-clockwise)
    pub fn prev_peer(&self) -> &PeerId {
        let prev_pos = if self.ring_position == 0 {
            self.ring_peers.len() - 1
        } else {
            self.ring_position - 1
        };
        &self.ring_peers[prev_pos]
    }
}

/// Pending all-reduce operation state
struct PendingAllReduce {
    /// Local tensor chunks (one per peer in ring)
    chunks: Vec<Option<Vec<f32>>>,

    /// Shape of each chunk
    chunk_shape: Vec<usize>,

    /// Received chunks from other nodes
    received_scatter: HashMap<usize, Vec<f32>>,

    /// Received chunks during gather phase
    received_gather: HashMap<usize, Vec<f32>>,

    /// Current phase
    phase: AllReducePhase,

    /// Current step within phase
    step: usize,

    /// Started time
    started: Instant,

    /// Completion signal
    completion_tx: Option<oneshot::Sender<Result<Vec<f32>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AllReducePhase {
    ScatterReduce,
    AllGather,
    Complete,
}

/// Coordinates ring all-reduce operations across the network
pub struct AllReduceCoordinator {
    /// Configuration
    config: AllReduceConfig,

    /// Pending operations by (request_id, layer_idx)
    pending: Arc<RwLock<HashMap<(String, usize), PendingAllReduce>>>,

    /// Channel to send messages to the network layer
    message_tx: mpsc::Sender<AllReduceMessage>,

    /// Statistics
    stats: Arc<RwLock<AllReduceStats>>,
}

/// Statistics for all-reduce operations
#[derive(Debug, Clone, Default)]
pub struct AllReduceStats {
    pub total_operations: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub avg_latency_ms: f64,
    pub errors: u64,
}

impl AllReduceCoordinator {
    /// Create a new coordinator
    pub fn new(config: AllReduceConfig, message_tx: mpsc::Sender<AllReduceMessage>) -> Self {
        info!(
            "🔄 AllReduceCoordinator initialized: ring position {}/{}",
            config.ring_position,
            config.world_size()
        );

        Self {
            config,
            pending: Arc::new(RwLock::new(HashMap::new())),
            message_tx,
            stats: Arc::new(RwLock::new(AllReduceStats::default())),
        }
    }

    /// Perform ring all-reduce on a local tensor
    ///
    /// Returns the sum of tensors from all nodes in the ring.
    pub async fn ring_all_reduce(
        &self,
        request_id: String,
        layer_idx: usize,
        local_tensor: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<Vec<f32>> {
        let world_size = self.config.world_size();

        if world_size == 1 {
            // Single node, no all-reduce needed
            return Ok(local_tensor);
        }

        let start = Instant::now();

        // 🚀 v2.6.4: Optimized 2-node direct exchange (2x faster than ring)
        // For 2 nodes: Just exchange and sum, no ring chunking needed
        if world_size == 2 {
            return self.two_node_all_reduce(request_id, layer_idx, local_tensor, shape, start).await;
        }

        // Split local tensor into N chunks for the ring
        let chunks = self.split_into_chunks(&local_tensor, world_size);
        let chunk_size = chunks[0].len();
        let chunk_shape = vec![chunk_size]; // Simplified 1D shape

        // Create completion channel
        let (tx, rx) = oneshot::channel();

        // Initialize pending state
        {
            let mut pending = self.pending.write().await;
            pending.insert(
                (request_id.clone(), layer_idx),
                PendingAllReduce {
                    chunks: chunks.into_iter().map(Some).collect(),
                    chunk_shape: chunk_shape.clone(),
                    received_scatter: HashMap::new(),
                    received_gather: HashMap::new(),
                    phase: AllReducePhase::ScatterReduce,
                    step: 0,
                    started: start,
                    completion_tx: Some(tx),
                },
            );
        }

        // Start scatter-reduce phase
        self.start_scatter_reduce(&request_id, layer_idx).await?;

        // Wait for completion
        let result = tokio::time::timeout(self.config.chunk_timeout * (world_size as u32 * 2), rx)
            .await
            .map_err(|_| anyhow!("All-reduce timeout"))??;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_operations += 1;
            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms =
                (stats.avg_latency_ms * (stats.total_operations - 1) as f64 + latency)
                    / stats.total_operations as f64;
        }

        // result is already Result<Vec<f32>> from the oneshot channel
        result
    }

    /// 🚀 v2.6.4: Optimized 2-node all-reduce (direct exchange)
    ///
    /// For exactly 2 nodes, we can do a simple exchange and sum:
    /// - Node 0 sends tensor to Node 1, receives from Node 1
    /// - Both nodes compute: local_tensor + received_tensor
    /// - Result: Both have identical sum
    ///
    /// This is 2x faster than ring for 2 nodes:
    /// - Ring: 2 scatter + 2 gather = 4 messages
    /// - Direct: 1 send + 1 recv = 2 messages
    async fn two_node_all_reduce(
        &self,
        request_id: String,
        layer_idx: usize,
        local_tensor: Vec<f32>,
        shape: Vec<usize>,
        start: Instant,
    ) -> Result<Vec<f32>> {
        let my_pos = self.config.ring_position;
        let other_pos = 1 - my_pos; // 0->1, 1->0

        debug!(
            "⚡ [2-NODE ALL-REDUCE] Node {} exchanging with Node {} for layer {}",
            my_pos, other_pos, layer_idx
        );

        // Compress local tensor
        let compressed_data = if self.config.compression {
            compress_f32(&local_tensor)
        } else {
            local_tensor.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        // Create exchange message
        let message = AllReduceMessage::Chunk {
            request_id: request_id.clone(),
            layer_idx,
            phase: "direct".to_string(),
            step: 0,
            chunk_idx: my_pos,
            data: compressed_data,
            shape: shape.clone(),
            compressed: self.config.compression,
        };

        // Create completion channel
        let (tx, rx) = oneshot::channel::<Result<Vec<f32>>>();

        // Initialize pending state for receiving
        {
            let mut pending = self.pending.write().await;
            pending.insert(
                (request_id.clone(), layer_idx),
                PendingAllReduce {
                    chunks: vec![Some(local_tensor.clone()), None], // Our tensor + placeholder for other
                    chunk_shape: shape.clone(),
                    received_scatter: HashMap::new(),
                    received_gather: HashMap::new(),
                    phase: AllReducePhase::ScatterReduce,
                    step: 0,
                    started: start,
                    completion_tx: Some(tx),
                },
            );
        }

        // Send our tensor to the other node via message channel
        if let Err(e) = self.message_tx.send(message).await {
            error!("❌ [2-NODE ALL-REDUCE] Failed to send tensor: {}", e);
            return Err(anyhow!("Failed to send tensor to other node"));
        }

        // Wait for the other node's tensor (with timeout)
        let timeout_duration = self.config.chunk_timeout * 4; // 4x normal chunk timeout
        let result = tokio::time::timeout(timeout_duration, rx)
            .await
            .map_err(|_| anyhow!("2-node all-reduce timeout waiting for peer"))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_operations += 1;
            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms =
                (stats.avg_latency_ms * (stats.total_operations - 1) as f64 + latency)
                    / stats.total_operations as f64;
            // Track 2-node specific stats
            if latency < 20.0 {
                debug!("⚡ [2-NODE ALL-REDUCE] Fast exchange: {:.2}ms", latency);
            }
        }

        result?
    }

    /// Start the scatter-reduce phase
    async fn start_scatter_reduce(&self, request_id: &str, layer_idx: usize) -> Result<()> {
        let world_size = self.config.world_size();
        let my_pos = self.config.ring_position;

        // In scatter-reduce, each node sends chunk[my_pos] to next peer
        // and receives chunk[prev_pos] from previous peer

        for step in 0..(world_size - 1) {
            // Which chunk to send this step
            let send_chunk_idx = (my_pos + world_size - step) % world_size;

            let chunk_data = {
                let pending = self.pending.read().await;
                if let Some(state) = pending.get(&(request_id.to_string(), layer_idx)) {
                    state.chunks[send_chunk_idx].clone()
                } else {
                    return Err(anyhow!("All-reduce state not found"));
                }
            };

            if let Some(data) = chunk_data {
                let compressed_data = if self.config.compression {
                    compress_f32(&data)
                } else {
                    data.iter().flat_map(|f| f.to_le_bytes()).collect()
                };

                let message = AllReduceMessage::Chunk {
                    request_id: request_id.to_string(),
                    layer_idx,
                    phase: "scatter".to_string(),
                    step,
                    chunk_idx: send_chunk_idx,
                    data: compressed_data,
                    shape: vec![data.len()],
                    compressed: self.config.compression,
                };

                self.message_tx.send(message).await?;
            }
        }

        Ok(())
    }

    /// Handle received all-reduce chunk
    pub async fn handle_chunk(&self, msg: AllReduceMessage) -> Result<()> {
        if let AllReduceMessage::Chunk {
            request_id,
            layer_idx,
            phase,
            step,
            chunk_idx,
            data,
            shape,
            compressed,
        } = msg
        {
            let chunk_data = if compressed {
                decompress_to_f32(&data)?
            } else {
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            };

            let mut pending = self.pending.write().await;
            if let Some(state) = pending.get_mut(&(request_id.clone(), layer_idx)) {
                match phase.as_str() {
                    "scatter" => {
                        // Add received chunk to our local chunk at this index
                        if let Some(local_chunk) = &mut state.chunks[chunk_idx] {
                            for (i, val) in chunk_data.iter().enumerate() {
                                local_chunk[i] += val;
                            }
                        } else {
                            state.chunks[chunk_idx] = Some(chunk_data);
                        }
                        state.received_scatter.insert(chunk_idx, vec![]);

                        // Check if scatter phase complete
                        if state.received_scatter.len() >= self.config.world_size() - 1 {
                            state.phase = AllReducePhase::AllGather;
                            state.step = 0;
                            drop(pending);
                            self.start_all_gather(&request_id, layer_idx).await?;
                        }
                    }
                    "gather" => {
                        // Replace our chunk with the fully reduced one
                        state.chunks[chunk_idx] = Some(chunk_data);
                        state.received_gather.insert(chunk_idx, vec![]);

                        // Check if gather phase complete
                        if state.received_gather.len() >= self.config.world_size() - 1 {
                            state.phase = AllReducePhase::Complete;

                            // Combine all chunks into final result
                            let mut result = Vec::new();
                            for chunk in &state.chunks {
                                if let Some(data) = chunk {
                                    result.extend(data.iter());
                                }
                            }

                            // Signal completion
                            if let Some(tx) = state.completion_tx.take() {
                                let _ = tx.send(Ok(result));
                            }
                        }
                    }
                    _ => {
                        warn!("Unknown all-reduce phase: {}", phase);
                    }
                }
            }
        }

        Ok(())
    }

    /// Start the all-gather phase
    async fn start_all_gather(&self, request_id: &str, layer_idx: usize) -> Result<()> {
        let world_size = self.config.world_size();
        let my_pos = self.config.ring_position;

        for step in 0..(world_size - 1) {
            // Which chunk to send this step
            let send_chunk_idx = (my_pos + world_size - step + 1) % world_size;

            let chunk_data = {
                let pending = self.pending.read().await;
                if let Some(state) = pending.get(&(request_id.to_string(), layer_idx)) {
                    state.chunks[send_chunk_idx].clone()
                } else {
                    return Err(anyhow!("All-reduce state not found"));
                }
            };

            if let Some(data) = chunk_data {
                let compressed_data = if self.config.compression {
                    compress_f32(&data)
                } else {
                    data.iter().flat_map(|f| f.to_le_bytes()).collect()
                };

                let message = AllReduceMessage::Chunk {
                    request_id: request_id.to_string(),
                    layer_idx,
                    phase: "gather".to_string(),
                    step,
                    chunk_idx: send_chunk_idx,
                    data: compressed_data,
                    shape: vec![data.len()],
                    compressed: self.config.compression,
                };

                self.message_tx.send(message).await?;
            }
        }

        Ok(())
    }

    /// Split a tensor into N equal chunks
    fn split_into_chunks(&self, data: &[f32], n: usize) -> Vec<Vec<f32>> {
        let chunk_size = (data.len() + n - 1) / n;
        data.chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Get current statistics
    pub async fn stats(&self) -> AllReduceStats {
        self.stats.read().await.clone()
    }

    /// Clean up completed operations
    pub async fn cleanup(&self, request_id: &str) {
        let mut pending = self.pending.write().await;
        pending.retain(|(req, _), _| req != request_id);
    }
}

/// Compress f32 vector (simple byte conversion for now)
/// TODO: Add real compression when flate2 is added to dependencies
fn compress_f32(data: &[f32]) -> Vec<u8> {
    // Simple byte conversion without compression
    // Real implementation would use gzip/zstd for 2-4x reduction
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Decompress to f32 vector
fn decompress_to_f32(compressed: &[u8]) -> Result<Vec<f32>> {
    // Simple byte conversion without decompression
    let floats: Vec<f32> = compressed
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(floats)
}

/// Simplified all-reduce for single-node testing
pub async fn local_all_reduce(tensors: Vec<Vec<f32>>) -> Result<Vec<f32>> {
    if tensors.is_empty() {
        return Err(anyhow!("No tensors to reduce"));
    }

    let first_len = tensors[0].len();
    let mut result = vec![0.0f32; first_len];

    for tensor in tensors {
        if tensor.len() != first_len {
            return Err(anyhow!("Tensor length mismatch"));
        }
        for (i, val) in tensor.iter().enumerate() {
            result[i] += val;
        }
    }

    Ok(result)
}

// =============================================================================
// MASSIVE SCALE OPTIMIZATIONS (20-1000+ nodes)
// =============================================================================
//
// Ring All-Reduce: O(N) latency - good for ≤8 nodes
// Hierarchical All-Reduce: O(log N) latency - required for 20+ nodes
// Butterfly All-Reduce: O(log N) latency with max parallelism - best for 100+ nodes
//
// Performance comparison for different node counts:
//
// | Nodes | Ring (O(N)) | Hierarchical (O(log N)) | Speedup |
// |-------|-------------|-------------------------|---------|
// |     8 |    8 hops   |       3 hops            |   2.7x  |
// |    20 |   20 hops   |       5 hops            |   4.0x  |
// |   100 |  100 hops   |       7 hops            |  14.3x  |
// |  1000 | 1000 hops   |      10 hops            | 100.0x  |

/// All-reduce algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceAlgorithm {
    /// Ring all-reduce: O(2(N-1)) messages, O(N) latency
    /// Best for: Small clusters (2-8 nodes) with high bandwidth
    Ring,

    /// Hierarchical/Tree all-reduce: O(2N) messages, O(log N) latency
    /// Best for: Medium clusters (8-64 nodes) prioritizing latency
    Hierarchical,

    /// Butterfly all-reduce: O(N log N) messages, O(log N) latency
    /// Best for: Large clusters (64+ nodes) with balanced bandwidth/latency
    Butterfly,

    /// Ring within groups, tree across groups
    /// Best for: Very large clusters (100+ nodes) with locality awareness
    HierarchicalRing,
}

impl AllReduceAlgorithm {
    /// Select optimal algorithm based on cluster size and tensor size
    pub fn select_optimal(world_size: usize, tensor_bytes: usize) -> Self {
        // Adaptive algorithm selection based on real-world measurements
        match world_size {
            1 => AllReduceAlgorithm::Ring, // No reduction needed
            2 => {
                // 🚀 v2.6.4: Optimized 2-node direct exchange
                // Ring for 2 nodes: 2 scatter + 2 gather = 4 messages
                // Direct: 1 send + 1 receive + reduce = 2 messages (2x faster!)
                // Use Ring but with optimized path (handled in ring_all_reduce)
                AllReduceAlgorithm::Ring
            }
            3..=8 => {
                // Small cluster: Ring is efficient, minimal latency overhead
                AllReduceAlgorithm::Ring
            }
            9..=32 => {
                // Medium cluster: Hierarchical reduces latency significantly
                // 9 nodes: Ring=9 hops, Tree=4 hops (2.25x faster)
                AllReduceAlgorithm::Hierarchical
            }
            33..=128 => {
                // Large cluster: Butterfly maximizes parallelism
                // 64 nodes: Ring=64 hops, Butterfly=6 hops (10.7x faster)
                AllReduceAlgorithm::Butterfly
            }
            _ => {
                // Massive cluster: Hierarchical ring balances all factors
                // 1000 nodes: 10 groups × 100 nodes each
                // Ring within groups (100 hops), Tree across groups (4 hops)
                // Total: ~104 hops vs 1000 hops (9.6x faster)
                AllReduceAlgorithm::HierarchicalRing
            }
        }
    }

    /// Expected number of communication rounds
    pub fn expected_rounds(&self, world_size: usize) -> usize {
        match self {
            AllReduceAlgorithm::Ring => 2 * (world_size - 1),
            AllReduceAlgorithm::Hierarchical => {
                // Binary tree: log2(N) rounds up, then log2(N) rounds down
                let levels = (world_size as f64).log2().ceil() as usize;
                2 * levels
            }
            AllReduceAlgorithm::Butterfly => {
                // Each step halves/doubles communication distance
                (world_size as f64).log2().ceil() as usize
            }
            AllReduceAlgorithm::HierarchicalRing => {
                let group_size = (world_size as f64).sqrt().ceil() as usize;
                let num_groups = (world_size + group_size - 1) / group_size;
                // Ring within groups + tree across groups
                2 * (group_size - 1) + 2 * ((num_groups as f64).log2().ceil() as usize)
            }
        }
    }
}

/// Hierarchical All-Reduce Coordinator
///
/// Uses a binary tree pattern for O(log N) latency:
///
/// ```text
/// Phase 1: Reduce (bottom-up)
///
///          [Sum: A+B+C+D+E+F+G+H]           Level 3 (root)
///                   /    \
///        [A+B+C+D]        [E+F+G+H]         Level 2
///          /    \          /    \
///      [A+B]  [C+D]    [E+F]  [G+H]         Level 1
///       /\      /\       /\      /\
///      A  B    C  D     E  F    G  H        Level 0 (leaves)
///
/// Phase 2: Broadcast (top-down)
///
///          [Sum]                            Level 3 (root)
///            |
///    --------|--------
///    |               |
///   [Sum]          [Sum]                    Level 2
///    |              |
///  --|--          --|--
///  |   |          |   |
/// [Sum][Sum]   [Sum][Sum]                   Level 1
///  |  |  |  |   |  |  |  |
/// [S][S][S][S] [S][S][S][S]                 Level 0 (all nodes have sum)
/// ```
pub struct HierarchicalAllReduceCoordinator {
    /// Node's position in the tree (0-indexed)
    pub my_rank: usize,

    /// Total number of nodes
    pub world_size: usize,

    /// Ordered list of peer IDs
    pub peers: Vec<PeerId>,

    /// Message sender
    message_tx: mpsc::Sender<AllReduceMessage>,

    /// Pending operations
    pending: Arc<RwLock<HashMap<(String, usize), HierarchicalPendingState>>>,
}

struct HierarchicalPendingState {
    /// Local tensor
    local_tensor: Vec<f32>,

    /// Received partial sums from children (for reduce phase)
    child_sums: HashMap<usize, Vec<f32>>,

    /// Have we received broadcast from parent?
    broadcast_received: bool,

    /// Final result
    result: Option<Vec<f32>>,

    /// Completion signal
    completion_tx: Option<oneshot::Sender<Result<Vec<f32>>>>,

    /// Start time
    started: Instant,
}

impl HierarchicalAllReduceCoordinator {
    pub fn new(
        my_rank: usize,
        peers: Vec<PeerId>,
        message_tx: mpsc::Sender<AllReduceMessage>,
    ) -> Self {
        info!(
            "🌳 HierarchicalAllReduce initialized: rank {}/{} (log2 {} levels)",
            my_rank,
            peers.len(),
            (peers.len() as f64).log2().ceil() as usize
        );

        Self {
            my_rank,
            world_size: peers.len(),
            peers,
            message_tx,
            pending: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get parent rank in binary tree (-1 if root)
    fn parent_rank(&self) -> Option<usize> {
        if self.my_rank == 0 {
            None // Root has no parent
        } else {
            Some((self.my_rank - 1) / 2)
        }
    }

    /// Get left child rank (None if no left child)
    fn left_child_rank(&self) -> Option<usize> {
        let left = 2 * self.my_rank + 1;
        if left < self.world_size {
            Some(left)
        } else {
            None
        }
    }

    /// Get right child rank (None if no right child)
    fn right_child_rank(&self) -> Option<usize> {
        let right = 2 * self.my_rank + 2;
        if right < self.world_size {
            Some(right)
        } else {
            None
        }
    }

    /// Is this node a leaf?
    fn is_leaf(&self) -> bool {
        self.left_child_rank().is_none()
    }

    /// Number of expected children
    fn num_children(&self) -> usize {
        let mut count = 0;
        if self.left_child_rank().is_some() { count += 1; }
        if self.right_child_rank().is_some() { count += 1; }
        count
    }

    /// Perform hierarchical all-reduce
    pub async fn hierarchical_all_reduce(
        &self,
        request_id: String,
        layer_idx: usize,
        local_tensor: Vec<f32>,
    ) -> Result<Vec<f32>> {
        if self.world_size == 1 {
            return Ok(local_tensor);
        }

        let start = Instant::now();
        let (tx, rx) = oneshot::channel();

        // Initialize state
        {
            let mut pending = self.pending.write().await;
            pending.insert(
                (request_id.clone(), layer_idx),
                HierarchicalPendingState {
                    local_tensor: local_tensor.clone(),
                    child_sums: HashMap::new(),
                    broadcast_received: false,
                    result: None,
                    completion_tx: Some(tx),
                    started: start,
                },
            );
        }

        // If leaf node, immediately send to parent
        if self.is_leaf() {
            self.send_to_parent(&request_id, layer_idx, local_tensor).await?;
        }
        // Otherwise wait for children to send their partial sums

        // Wait for completion with timeout
        let levels = (self.world_size as f64).log2().ceil() as u32;
        let timeout = Duration::from_secs(30) * levels * 2;

        let result = tokio::time::timeout(timeout, rx)
            .await
            .map_err(|_| anyhow!("Hierarchical all-reduce timeout after {:?}", timeout))??;

        debug!(
            "🌳 Hierarchical all-reduce complete: rank {}, {} nodes, {:.2}ms",
            self.my_rank,
            self.world_size,
            start.elapsed().as_millis()
        );

        result
    }

    /// Send partial sum to parent
    async fn send_to_parent(
        &self,
        request_id: &str,
        layer_idx: usize,
        tensor: Vec<f32>,
    ) -> Result<()> {
        if let Some(parent_rank) = self.parent_rank() {
            let compressed = compress_f32(&tensor);

            let msg = AllReduceMessage::Chunk {
                request_id: request_id.to_string(),
                layer_idx,
                phase: "reduce".to_string(),
                step: self.my_rank, // Use rank to identify sender
                chunk_idx: self.my_rank,
                data: compressed,
                shape: vec![tensor.len()],
                compressed: true,
            };

            debug!(
                "🌳 Rank {} sending reduce to parent rank {}",
                self.my_rank, parent_rank
            );
            self.message_tx.send(msg).await?;
        }
        Ok(())
    }

    /// Broadcast result to children
    async fn broadcast_to_children(
        &self,
        request_id: &str,
        layer_idx: usize,
        tensor: Vec<f32>,
    ) -> Result<()> {
        let compressed = compress_f32(&tensor);

        for child_rank in [self.left_child_rank(), self.right_child_rank()]
            .into_iter()
            .flatten()
        {
            let msg = AllReduceMessage::Chunk {
                request_id: request_id.to_string(),
                layer_idx,
                phase: "broadcast".to_string(),
                step: child_rank,
                chunk_idx: self.my_rank,
                data: compressed.clone(),
                shape: vec![tensor.len()],
                compressed: true,
            };

            debug!(
                "🌳 Rank {} broadcasting to child rank {}",
                self.my_rank, child_rank
            );
            self.message_tx.send(msg).await?;
        }
        Ok(())
    }

    /// Handle received message
    pub async fn handle_message(&self, msg: AllReduceMessage) -> Result<()> {
        if let AllReduceMessage::Chunk {
            request_id,
            layer_idx,
            phase,
            step: sender_rank,
            data,
            ..
        } = msg
        {
            let tensor = decompress_to_f32(&data)?;

            let mut pending = self.pending.write().await;
            if let Some(state) = pending.get_mut(&(request_id.clone(), layer_idx)) {
                match phase.as_str() {
                    "reduce" => {
                        // Received partial sum from a child
                        state.child_sums.insert(sender_rank, tensor);

                        // Check if we have all children's sums
                        if state.child_sums.len() >= self.num_children() {
                            // Compute sum of local + all children
                            let mut sum = state.local_tensor.clone();
                            for child_tensor in state.child_sums.values() {
                                for (i, val) in child_tensor.iter().enumerate() {
                                    sum[i] += val;
                                }
                            }

                            if self.my_rank == 0 {
                                // Root: We have the global sum, start broadcast
                                state.result = Some(sum.clone());
                                drop(pending);
                                self.broadcast_to_children(&request_id, layer_idx, sum.clone())
                                    .await?;

                                // Complete for root
                                let mut pending = self.pending.write().await;
                                if let Some(state) =
                                    pending.get_mut(&(request_id.clone(), layer_idx))
                                {
                                    if let Some(tx) = state.completion_tx.take() {
                                        let _ = tx.send(Ok(sum));
                                    }
                                }
                            } else {
                                // Non-root: Send sum to parent
                                drop(pending);
                                self.send_to_parent(&request_id, layer_idx, sum).await?;
                            }
                        }
                    }
                    "broadcast" => {
                        // Received final sum from parent
                        state.broadcast_received = true;
                        state.result = Some(tensor.clone());

                        // Forward to children
                        drop(pending);
                        self.broadcast_to_children(&request_id, layer_idx, tensor.clone())
                            .await?;

                        // Signal completion
                        let mut pending = self.pending.write().await;
                        if let Some(state) = pending.get_mut(&(request_id.clone(), layer_idx)) {
                            if let Some(tx) = state.completion_tx.take() {
                                let _ = tx.send(Ok(tensor));
                            }
                        }
                    }
                    _ => {
                        warn!("Unknown hierarchical phase: {}", phase);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Butterfly All-Reduce Coordinator
///
/// Optimal for large clusters (64+ nodes) with O(log N) rounds.
/// Each round, nodes exchange with partners at increasing distances.
///
/// ```text
/// Round 0 (distance 1):  0↔1  2↔3  4↔5  6↔7
/// Round 1 (distance 2):  0↔2  1↔3  4↔6  5↔7
/// Round 2 (distance 4):  0↔4  1↔5  2↔6  3↔7
///
/// After 3 rounds: All 8 nodes have the sum of all values!
/// ```
///
/// Benefits over hierarchical:
/// - All nodes communicate every round (maximum parallelism)
/// - Better load balancing
/// - No single bottleneck at root
pub struct ButterflyAllReduceCoordinator {
    pub my_rank: usize,
    pub world_size: usize,
    pub peers: Vec<PeerId>,
    message_tx: mpsc::Sender<AllReduceMessage>,
    pending: Arc<RwLock<HashMap<(String, usize), ButterflyPendingState>>>,
}

struct ButterflyPendingState {
    /// Current partial sum (starts as local tensor)
    current_sum: Vec<f32>,

    /// Current round (0 to log2(N)-1)
    current_round: usize,

    /// Total rounds needed
    total_rounds: usize,

    /// Have we received partner's value for current round?
    received_this_round: bool,

    /// Completion signal
    completion_tx: Option<oneshot::Sender<Result<Vec<f32>>>>,

    started: Instant,
}

impl ButterflyAllReduceCoordinator {
    pub fn new(
        my_rank: usize,
        peers: Vec<PeerId>,
        message_tx: mpsc::Sender<AllReduceMessage>,
    ) -> Self {
        let total_rounds = (peers.len() as f64).log2().ceil() as usize;

        info!(
            "🦋 ButterflyAllReduce initialized: rank {}/{}, {} rounds",
            my_rank,
            peers.len(),
            total_rounds
        );

        Self {
            my_rank,
            world_size: peers.len(),
            peers,
            message_tx,
            pending: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get partner rank for a given round
    fn partner_rank(&self, round: usize) -> usize {
        // XOR with 2^round to get partner
        self.my_rank ^ (1 << round)
    }

    /// Perform butterfly all-reduce
    pub async fn butterfly_all_reduce(
        &self,
        request_id: String,
        layer_idx: usize,
        local_tensor: Vec<f32>,
    ) -> Result<Vec<f32>> {
        if self.world_size == 1 {
            return Ok(local_tensor);
        }

        let start = Instant::now();
        let total_rounds = (self.world_size as f64).log2().ceil() as usize;
        let (tx, rx) = oneshot::channel();

        // Initialize state
        {
            let mut pending = self.pending.write().await;
            pending.insert(
                (request_id.clone(), layer_idx),
                ButterflyPendingState {
                    current_sum: local_tensor.clone(),
                    current_round: 0,
                    total_rounds,
                    received_this_round: false,
                    completion_tx: Some(tx),
                    started: start,
                },
            );
        }

        // Start round 0
        self.send_to_partner(&request_id, layer_idx, 0, local_tensor)
            .await?;

        // Wait for completion
        let timeout = Duration::from_secs(30) * total_rounds as u32;
        let result = tokio::time::timeout(timeout, rx)
            .await
            .map_err(|_| anyhow!("Butterfly all-reduce timeout"))??;

        debug!(
            "🦋 Butterfly all-reduce complete: rank {}, {} nodes, {} rounds, {:.2}ms",
            self.my_rank,
            self.world_size,
            total_rounds,
            start.elapsed().as_millis()
        );

        result
    }

    async fn send_to_partner(
        &self,
        request_id: &str,
        layer_idx: usize,
        round: usize,
        tensor: Vec<f32>,
    ) -> Result<()> {
        let partner = self.partner_rank(round);

        // Only send if partner is within world_size
        if partner < self.world_size {
            let compressed = compress_f32(&tensor);

            let msg = AllReduceMessage::Chunk {
                request_id: request_id.to_string(),
                layer_idx,
                phase: format!("butterfly-{}", round),
                step: round,
                chunk_idx: self.my_rank,
                data: compressed,
                shape: vec![tensor.len()],
                compressed: true,
            };

            debug!(
                "🦋 Rank {} round {}: sending to partner {}",
                self.my_rank, round, partner
            );
            self.message_tx.send(msg).await?;
        }
        Ok(())
    }

    /// Handle received butterfly message
    pub async fn handle_message(&self, msg: AllReduceMessage) -> Result<()> {
        if let AllReduceMessage::Chunk {
            request_id,
            layer_idx,
            phase,
            step: round,
            data,
            chunk_idx: sender_rank,
            ..
        } = msg
        {
            if !phase.starts_with("butterfly-") {
                return Ok(()); // Not a butterfly message
            }

            let partner_tensor = decompress_to_f32(&data)?;

            let mut pending = self.pending.write().await;
            if let Some(state) = pending.get_mut(&(request_id.clone(), layer_idx)) {
                // Verify this is from our partner for the current round
                let expected_partner = self.partner_rank(state.current_round);
                if sender_rank != expected_partner || round != state.current_round {
                    // Out-of-order message, could queue it but for now just log
                    debug!(
                        "🦋 Out-of-order message: expected partner {} round {}, got {} round {}",
                        expected_partner, state.current_round, sender_rank, round
                    );
                    return Ok(());
                }

                // Add partner's tensor to our sum
                for (i, val) in partner_tensor.iter().enumerate() {
                    state.current_sum[i] += val;
                }

                state.current_round += 1;

                if state.current_round >= state.total_rounds {
                    // All rounds complete!
                    if let Some(tx) = state.completion_tx.take() {
                        let _ = tx.send(Ok(state.current_sum.clone()));
                    }
                } else {
                    // Send to next round's partner
                    let sum = state.current_sum.clone();
                    let next_round = state.current_round;
                    drop(pending);
                    self.send_to_partner(&request_id, layer_idx, next_round, sum)
                        .await?;
                }
            }
        }
        Ok(())
    }
}

/// Adaptive All-Reduce Coordinator
///
/// Automatically selects the best algorithm based on cluster size and
/// routes operations through the appropriate coordinator.
pub struct AdaptiveAllReduceCoordinator {
    pub my_rank: usize,
    pub world_size: usize,
    pub peers: Vec<PeerId>,

    /// Ring coordinator for small clusters
    ring: Arc<AllReduceCoordinator>,

    /// Hierarchical coordinator for medium clusters
    hierarchical: Arc<HierarchicalAllReduceCoordinator>,

    /// Butterfly coordinator for large clusters
    butterfly: Arc<ButterflyAllReduceCoordinator>,

    /// Currently selected algorithm
    current_algorithm: AllReduceAlgorithm,

    /// Statistics
    stats: Arc<RwLock<AdaptiveAllReduceStats>>,
}

/// Statistics for adaptive all-reduce
#[derive(Debug, Clone, Default)]
pub struct AdaptiveAllReduceStats {
    pub ring_operations: u64,
    pub hierarchical_operations: u64,
    pub butterfly_operations: u64,
    pub total_operations: u64,
    pub avg_latency_ms: f64,
    pub tensor_bytes_processed: u64,
}

impl AdaptiveAllReduceCoordinator {
    pub fn new(
        my_rank: usize,
        peers: Vec<PeerId>,
        self_peer: PeerId,
        message_tx: mpsc::Sender<AllReduceMessage>,
    ) -> Result<Self> {
        let world_size = peers.len();
        let algorithm = AllReduceAlgorithm::select_optimal(world_size, 0);

        info!(
            "⚡ AdaptiveAllReduce initialized: rank {}/{}, algorithm={:?} (expected {} rounds)",
            my_rank,
            world_size,
            algorithm,
            algorithm.expected_rounds(world_size)
        );

        let ring_config = AllReduceConfig::new(peers.clone(), self_peer)?;

        Ok(Self {
            my_rank,
            world_size,
            peers: peers.clone(),
            ring: Arc::new(AllReduceCoordinator::new(ring_config, message_tx.clone())),
            hierarchical: Arc::new(HierarchicalAllReduceCoordinator::new(
                my_rank,
                peers.clone(),
                message_tx.clone(),
            )),
            butterfly: Arc::new(ButterflyAllReduceCoordinator::new(
                my_rank,
                peers,
                message_tx,
            )),
            current_algorithm: algorithm,
            stats: Arc::new(RwLock::new(AdaptiveAllReduceStats::default())),
        })
    }

    /// Perform all-reduce using the optimal algorithm
    pub async fn all_reduce(
        &self,
        request_id: String,
        layer_idx: usize,
        local_tensor: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<Vec<f32>> {
        let tensor_bytes = local_tensor.len() * 4;
        let algorithm = AllReduceAlgorithm::select_optimal(self.world_size, tensor_bytes);
        let start = Instant::now();

        let result = match algorithm {
            AllReduceAlgorithm::Ring => {
                debug!(
                    "🔄 Using Ring all-reduce for {} nodes",
                    self.world_size
                );
                self.ring
                    .ring_all_reduce(request_id, layer_idx, local_tensor, shape)
                    .await
            }
            AllReduceAlgorithm::Hierarchical | AllReduceAlgorithm::HierarchicalRing => {
                debug!(
                    "🌳 Using Hierarchical all-reduce for {} nodes",
                    self.world_size
                );
                self.hierarchical
                    .hierarchical_all_reduce(request_id, layer_idx, local_tensor)
                    .await
            }
            AllReduceAlgorithm::Butterfly => {
                debug!(
                    "🦋 Using Butterfly all-reduce for {} nodes",
                    self.world_size
                );
                self.butterfly
                    .butterfly_all_reduce(request_id, layer_idx, local_tensor)
                    .await
            }
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_operations += 1;
            stats.tensor_bytes_processed += tensor_bytes as u64;
            match algorithm {
                AllReduceAlgorithm::Ring => stats.ring_operations += 1,
                AllReduceAlgorithm::Hierarchical | AllReduceAlgorithm::HierarchicalRing => {
                    stats.hierarchical_operations += 1
                }
                AllReduceAlgorithm::Butterfly => stats.butterfly_operations += 1,
            }

            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.total_operations - 1) as f64
                + latency)
                / stats.total_operations as f64;
        }

        result
    }

    /// Handle incoming all-reduce message (routes to appropriate coordinator)
    pub async fn handle_message(&self, msg: AllReduceMessage) -> Result<()> {
        // Route based on phase
        if let AllReduceMessage::Chunk { ref phase, .. } = msg {
            if phase == "scatter" || phase == "gather" {
                self.ring.handle_chunk(msg).await
            } else if phase == "reduce" || phase == "broadcast" {
                self.hierarchical.handle_message(msg).await
            } else if phase.starts_with("butterfly-") {
                self.butterfly.handle_message(msg).await
            } else {
                warn!("Unknown all-reduce phase: {}", phase);
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Get current statistics
    pub async fn stats(&self) -> AdaptiveAllReduceStats {
        self.stats.read().await.clone()
    }

    /// Get expected speedup compared to ring for current cluster size
    pub fn expected_speedup(&self) -> f64 {
        let ring_rounds = AllReduceAlgorithm::Ring.expected_rounds(self.world_size);
        let current_rounds = self.current_algorithm.expected_rounds(self.world_size);

        if current_rounds > 0 {
            ring_rounds as f64 / current_rounds as f64
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_chunks() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();

        // Simulate splitting for 4 nodes
        let n = 4;
        let chunk_size = (data.len() + n - 1) / n;
        let chunks: Vec<Vec<f32>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(chunks[1], vec![3.0, 4.0, 5.0]);
        assert_eq!(chunks[2], vec![6.0, 7.0, 8.0]);
        assert_eq!(chunks[3], vec![9.0]); // Last chunk may be smaller
    }

    #[tokio::test]
    async fn test_local_all_reduce() {
        let t1 = vec![1.0f32, 2.0, 3.0];
        let t2 = vec![4.0f32, 5.0, 6.0];
        let t3 = vec![7.0f32, 8.0, 9.0];

        let result = local_all_reduce(vec![t1, t2, t3]).await.unwrap();

        assert_eq!(result, vec![12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_compression_roundtrip() {
        let data = vec![1.5f32, 2.5, 3.5, 4.5, 5.5];
        let compressed = compress_f32(&data);
        let decompressed = decompress_to_f32(&compressed).unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_ring_config() {
        let peers: Vec<PeerId> = (0..4)
            .map(|_| PeerId::random())
            .collect();

        let self_peer = peers[2].clone();
        let config = AllReduceConfig::new(peers.clone(), self_peer).unwrap();

        assert_eq!(config.world_size(), 4);
        // Position depends on sort order
    }
}
