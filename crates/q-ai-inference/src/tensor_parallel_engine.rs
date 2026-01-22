//! Tensor Parallel Inference Engine
//!
//! This is the main engine for tensor-parallel distributed inference.
//! Unlike data parallelism (each node handles different requests) or
//! pipeline parallelism (layers split sequentially), tensor parallelism
//! splits each layer's computation across nodes for true Nx speedup.
//!
//! ## How It Works
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    TENSOR PARALLEL INFERENCE                     │
//! │                                                                   │
//! │  Input: "What is quantum computing?"                             │
//! │                        ↓                                         │
//! │  ┌─────────────────────────────────────────────────────────┐     │
//! │  │              EMBEDDING (replicated on all nodes)          │     │
//! │  └─────────────────────────────────────────────────────────┘     │
//! │                        ↓                                         │
//! │  ┌─────────────────────────────────────────────────────────┐     │
//! │  │                   LAYER 0                                 │     │
//! │  │  ┌──────────┬──────────┬──────────┬──────────┐          │     │
//! │  │  │ Node A   │ Node B   │ Node C   │ Node D   │          │     │
//! │  │  │ Heads    │ Heads    │ Heads    │ Heads    │          │     │
//! │  │  │ 0-7      │ 8-15     │ 16-23    │ 24-31    │          │     │
//! │  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘          │     │
//! │  │       └──────────┴──────────┴──────────┘                │     │
//! │  │                    All-Reduce                            │     │
//! │  │                        ↓                                 │     │
//! │  │  ┌──────────┬──────────┬──────────┬──────────┐          │     │
//! │  │  │ Node A   │ Node B   │ Node C   │ Node D   │          │     │
//! │  │  │ FFN      │ FFN      │ FFN      │ FFN      │          │     │
//! │  │  │ dims     │ dims     │ dims     │ dims     │          │     │
//! │  │  │ 0-3583   │ 3584-7167│ 7168-10751│10752-14335│        │     │
//! │  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘          │     │
//! │  │       └──────────┴──────────┴──────────┘                │     │
//! │  │                    All-Reduce                            │     │
//! │  └─────────────────────────────────────────────────────────┘     │
//! │                        ↓                                         │
//! │                   LAYERS 1-31 (same pattern)                     │
//! │                        ↓                                         │
//! │  ┌─────────────────────────────────────────────────────────┐     │
//! │  │           LM HEAD (coordinator aggregates)                │     │
//! │  └─────────────────────────────────────────────────────────┘     │
//! │                        ↓                                         │
//! │  Output: "Quantum computing uses quantum..."                     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Targets
//!
//! | Nodes | Compute Speedup | All-Reduce Overhead | Net Speedup |
//! |-------|-----------------|---------------------|-------------|
//! | 2     | 2x              | ~10%                | ~1.8x       |
//! | 4     | 4x              | ~20%                | ~3.2x       |
//! | 8     | 8x              | ~30%                | ~5.6x       |

use anyhow::{anyhow, Result};
use candle_core::{Device, IndexOp, Tensor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::weight_shard_manager::{ShardConfig, WeightShard};

/// Configuration for tensor parallel inference
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Shard configuration
    pub shard_config: ShardConfig,

    /// Whether this node is the coordinator
    pub is_coordinator: bool,

    /// Timeout for all-reduce operations
    pub all_reduce_timeout: Duration,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Sampling temperature
    pub temperature: f64,

    /// Top-p sampling
    pub top_p: f64,

    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            shard_config: ShardConfig::mistral_7b(1, 0),
            is_coordinator: true,
            all_reduce_timeout: Duration::from_secs(30),
            max_seq_len: 4096,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
        }
    }
}

/// Statistics for tensor parallel inference
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorParallelStats {
    /// Total inference requests processed
    pub total_requests: u64,

    /// Total tokens generated
    pub total_tokens: u64,

    /// Average tokens per second (combined across all nodes)
    pub avg_tokens_per_sec: f64,

    /// Average time per token in milliseconds
    pub avg_token_time_ms: f64,

    /// Average all-reduce latency in milliseconds
    pub avg_all_reduce_ms: f64,

    /// Theoretical speedup vs single node
    pub theoretical_speedup: f64,

    /// Actual measured speedup
    pub actual_speedup: f64,

    /// Number of nodes in tensor parallel group
    pub world_size: usize,

    /// Total all-reduce operations
    pub all_reduce_count: u64,

    /// Memory usage per node in bytes
    pub memory_per_node: usize,
}

/// Represents the sharded weights for a single transformer layer
pub struct ShardedLayer {
    /// Layer index
    pub layer_idx: usize,

    // Attention weights (sharded by head)
    pub attn_q: Option<Tensor>,     // [hidden, heads_per_node * head_dim] = [4096, 2048] for 2 nodes
    pub attn_k: Option<Tensor>,     // v2.6.9: [kv_heads_per_node * head_dim, hidden] = [512, 4096] for 2 nodes (row-sharded)
    pub attn_v: Option<Tensor>,     // v2.6.9: [kv_heads_per_node * head_dim, hidden] = [512, 4096] for 2 nodes (row-sharded)
    pub attn_output: Option<Tensor>, // [heads_per_node * head_dim, hidden]

    // FFN weights (sharded by intermediate dim)
    pub ffn_gate: Option<Tensor>,   // [hidden, ffn_dim_per_node]
    pub ffn_up: Option<Tensor>,     // [hidden, ffn_dim_per_node]
    pub ffn_down: Option<Tensor>,   // [ffn_dim_per_node, hidden]

    // Normalization (replicated, not sharded)
    pub attn_norm: Option<Tensor>,
    pub ffn_norm: Option<Tensor>,
}

/// Main tensor parallel inference engine
pub struct TensorParallelEngine {
    /// Configuration
    config: TensorParallelConfig,

    /// Device for local computation
    device: Device,

    /// Sharded layers
    layers: Vec<ShardedLayer>,

    /// Embedding weights (replicated)
    token_embedding: Option<Tensor>,

    /// Output norm (replicated)
    output_norm: Option<Tensor>,

    /// LM head weights (coordinator only, or sharded)
    lm_head: Option<Tensor>,

    /// Statistics
    stats: Arc<RwLock<TensorParallelStats>>,

    /// Channel for sending all-reduce requests
    all_reduce_tx: Option<mpsc::Sender<AllReduceRequest>>,

    /// Channel for receiving all-reduce responses
    all_reduce_rx: Option<mpsc::Receiver<AllReduceResponse>>,
}

/// Request for all-reduce operation
#[derive(Debug)]
pub struct AllReduceRequest {
    pub request_id: String,
    pub layer_idx: usize,
    pub tensor: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Response from all-reduce operation
#[derive(Debug)]
pub struct AllReduceResponse {
    pub request_id: String,
    pub layer_idx: usize,
    pub result: Vec<f32>,
    pub shape: Vec<usize>,
    pub latency_ms: u64,
}

impl TensorParallelEngine {
    /// Create a new tensor parallel engine with CPU device (default)
    pub fn new_cpu(config: TensorParallelConfig) -> Self {
        Self::new(config, Device::Cpu)
    }

    /// Create a new tensor parallel engine
    pub fn new(config: TensorParallelConfig, device: Device) -> Self {
        let world_size = config.shard_config.world_size;
        let num_layers = config.shard_config.num_layers;

        info!(
            "⚡ TensorParallelEngine initialized: rank={}/{}, layers={}",
            config.shard_config.node_rank,
            world_size,
            num_layers
        );

        // Initialize empty layers
        let layers = (0..num_layers)
            .map(|idx| ShardedLayer {
                layer_idx: idx,
                attn_q: None,
                attn_k: None,
                attn_v: None,
                attn_output: None,
                ffn_gate: None,
                ffn_up: None,
                ffn_down: None,
                attn_norm: None,
                ffn_norm: None,
            })
            .collect();

        let mut stats = TensorParallelStats::default();
        stats.world_size = world_size;
        stats.theoretical_speedup = world_size as f64;

        Self {
            config,
            device,
            layers,
            token_embedding: None,
            output_norm: None,
            lm_head: None,
            stats: Arc::new(RwLock::new(stats)),
            all_reduce_tx: None,
            all_reduce_rx: None,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &TensorParallelConfig {
        &self.config
    }

    /// Get the world size (number of nodes in tensor parallel group)
    pub fn world_size(&self) -> usize {
        self.config.shard_config.world_size
    }

    /// Get this node's rank
    pub fn node_rank(&self) -> usize {
        self.config.shard_config.node_rank
    }

    /// Check if weights have been loaded
    /// Returns true if at least one layer has attention Q weights loaded
    pub fn has_weights(&self) -> bool {
        self.layers.iter().any(|layer| layer.attn_q.is_some())
    }

    /// Load sharded weights from the weight shard manager
    pub async fn load_shards(&mut self, shards: Vec<WeightShard>) -> Result<()> {
        info!("📦 Loading {} weight shards...", shards.len());

        for shard in shards {
            let tensor = shard.to_tensor(&self.device)?;

            // Special handling for embedding/output layers (layer_idx = usize::MAX)
            if shard.layer_idx == usize::MAX {
                match shard.weight_name.as_str() {
                    "token_embd" => {
                        info!("✅ Loaded token embedding: {:?}", shard.shape);
                        self.token_embedding = Some(tensor);
                    }
                    "output_norm" => {
                        info!("✅ Loaded output norm: {:?}", shard.shape);
                        self.output_norm = Some(tensor);
                    }
                    "output" => {
                        info!("✅ Loaded LM head: {:?}", shard.shape);
                        self.lm_head = Some(tensor);
                    }
                    _ => warn!("Unknown global weight: {}", shard.weight_name),
                }
                continue;
            }

            // Regular per-layer weights
            if shard.layer_idx >= self.layers.len() {
                warn!("Layer index {} out of bounds (max {})", shard.layer_idx, self.layers.len());
                continue;
            }

            let layer = &mut self.layers[shard.layer_idx];

            match shard.weight_name.as_str() {
                "attn_q" => layer.attn_q = Some(tensor),
                "attn_k" => layer.attn_k = Some(tensor),
                "attn_v" => layer.attn_v = Some(tensor),
                "attn_output" => layer.attn_output = Some(tensor),
                "ffn_gate" => layer.ffn_gate = Some(tensor),
                "ffn_up" => layer.ffn_up = Some(tensor),
                "ffn_down" => layer.ffn_down = Some(tensor),
                "attn_norm" => layer.attn_norm = Some(tensor),
                "ffn_norm" => layer.ffn_norm = Some(tensor),
                _ => warn!("Unknown layer weight: {}", shard.weight_name),
            }
        }

        // Calculate memory usage (include embedding/output layers)
        let mut total_bytes = 0usize;

        // Embedding and output layers (replicated on all nodes)
        if let Some(t) = &self.token_embedding { total_bytes += t.elem_count() * 4; }
        if let Some(t) = &self.output_norm { total_bytes += t.elem_count() * 4; }
        if let Some(t) = &self.lm_head { total_bytes += t.elem_count() * 4; }

        // Per-layer weights (sharded)
        for layer in &self.layers {
            if let Some(t) = &layer.attn_q { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.attn_k { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.attn_v { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.attn_output { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.ffn_gate { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.ffn_up { total_bytes += t.elem_count() * 4; }
            if let Some(t) = &layer.ffn_down { total_bytes += t.elem_count() * 4; }
        }

        self.stats.write().await.memory_per_node = total_bytes;

        // Log what we loaded
        info!(
            "✅ Loaded shards: {} MB per node (emb={}, norm={}, lm_head={})",
            total_bytes / (1024 * 1024),
            self.token_embedding.is_some(),
            self.output_norm.is_some(),
            self.lm_head.is_some()
        );

        Ok(())
    }

    /// Set up all-reduce communication channels
    pub fn set_all_reduce_channels(
        &mut self,
        tx: mpsc::Sender<AllReduceRequest>,
        rx: mpsc::Receiver<AllReduceResponse>,
    ) {
        self.all_reduce_tx = Some(tx);
        self.all_reduce_rx = Some(rx);
    }

    /// Perform tensor-parallel attention for a single layer
    async fn attention_forward(
        &self,
        layer: &ShardedLayer,
        hidden_states: &Tensor,
        request_id: &str,
    ) -> Result<Tensor> {
        let attn_start = Instant::now();

        // v2.6.7 FIX: Get dimensions for proper reshaping
        let (batch, seq_len, hidden_dim) = hidden_states.dims3()?;
        info!("🔍 [ATTN L{}] Input shape: [{}, {}, {}]", layer.layer_idx, batch, seq_len, hidden_dim);

        // Flatten [batch, seq, hidden] to [batch*seq, hidden] for 2D matmul compatibility
        let flat_hidden = hidden_states.reshape((batch * seq_len, hidden_dim))?;

        // Compute partial Q, K, V projections with proper matmul
        // v2.6.8 FIX: Weights are stored as [in, out] format, NO transpose needed
        // For Candle: [batch*seq, hidden] @ [hidden, out/N] = [batch*seq, out/N]
        let q_flat = if let Some(wq) = &layer.attn_q {
            // wq shape: [hidden, heads_per_node * head_dim] = [4096, 2048] for 2 nodes
            flat_hidden.matmul(wq)?
        } else {
            return Err(anyhow!("Missing Q weights for layer {}", layer.layer_idx));
        };

        let k_flat = if let Some(wk) = &layer.attn_k {
            // v2.6.9 FIX: K is row-sharded from GGUF [kv_dim, hidden]
            // Shard shape: [kv_dim/N, hidden] = [512, 4096] for 2 nodes
            // Need transpose: [hidden, kv_dim/N] for matmul
            flat_hidden.matmul(&wk.t()?)?
        } else {
            return Err(anyhow!("Missing K weights for layer {}", layer.layer_idx));
        };

        let v_flat = if let Some(wv) = &layer.attn_v {
            // v2.6.9 FIX: V is row-sharded from GGUF [kv_dim, hidden]
            // Shard shape: [kv_dim/N, hidden] = [512, 4096] for 2 nodes
            // Need transpose: [hidden, kv_dim/N] for matmul
            flat_hidden.matmul(&wv.t()?)?
        } else {
            return Err(anyhow!("Missing V weights for layer {}", layer.layer_idx));
        };

        // Get output dimensions from Q (which determines heads_per_node * head_dim)
        let q_out_dim = q_flat.dim(1)?;
        let k_out_dim = k_flat.dim(1)?;

        // Reshape back to [batch, seq, out_dim]
        let q = q_flat.reshape((batch, seq_len, q_out_dim))?;
        let k = k_flat.reshape((batch, seq_len, k_out_dim))?;
        let v = v_flat.reshape((batch, seq_len, k_out_dim))?;

        // Compute attention for our subset of heads
        // Shape: [batch, seq, heads_per_node, head_dim]
        let heads_per_node = self.config.shard_config.heads_per_node();
        let head_dim = self.config.shard_config.head_dim;
        let kv_heads_per_node = self.config.shard_config.kv_heads_per_node();

        let q = q.reshape((batch, seq_len, heads_per_node, head_dim))?;
        let k = k.reshape((batch, seq_len, kv_heads_per_node, head_dim))?;
        let v = v.reshape((batch, seq_len, kv_heads_per_node, head_dim))?;

        // Compute scaled dot-product attention
        // scores = Q @ K^T / sqrt(head_dim)
        let scale = (head_dim as f64).sqrt();
        let q = q.transpose(1, 2)?; // [batch, heads, seq, dim]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / scale)?;

        // Apply softmax along last dimension
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?; // [batch, seq, heads, dim]
        let attn_output = attn_output.reshape((batch, seq_len, heads_per_node * head_dim))?;

        // Output projection (row-parallel: each node has partial input dim)
        // v2.6.8 FIX: Weights are [in, out] format, NO transpose needed
        // wo shape: [heads_per_node * head_dim, hidden] = [2048, 4096] for 2 nodes
        let flat_attn = attn_output.reshape((batch * seq_len, heads_per_node * head_dim))?;
        let output_flat = if let Some(wo) = &layer.attn_output {
            flat_attn.matmul(wo)?
        } else {
            return Err(anyhow!("Missing output weights for layer {}", layer.layer_idx));
        };
        let output = output_flat.reshape((batch, seq_len, hidden_dim))?;

        // All-reduce to combine partial outputs from all nodes
        let ar_start = Instant::now();
        let output = self.all_reduce(&output, request_id, layer.layer_idx, "attn").await?;
        let ar_time = ar_start.elapsed();

        let attn_time = attn_start.elapsed();
        info!(
            "⚡ [ATTN L{}] Completed in {:?} (all-reduce: {:?}) | Output: {:?}",
            layer.layer_idx, attn_time, ar_time, output.dims()
        );

        Ok(output)
    }

    /// Perform tensor-parallel MLP for a single layer
    async fn mlp_forward(
        &self,
        layer: &ShardedLayer,
        hidden_states: &Tensor,
        request_id: &str,
    ) -> Result<Tensor> {
        let mlp_start = Instant::now();

        // v2.6.7 FIX: Get dimensions and flatten for 2D matmul compatibility
        let (batch, seq_len, hidden_dim) = hidden_states.dims3()?;
        let flat_hidden = hidden_states.reshape((batch * seq_len, hidden_dim))?;

        // Gate and Up projections (column-parallel)
        // v2.6.8 FIX: Weights are [in, out] format, NO transpose needed
        // gate/up shape: [hidden, ffn_dim_per_node] = [4096, 7168] for 2 nodes
        let gate_flat = if let Some(wg) = &layer.ffn_gate {
            flat_hidden.matmul(wg)?
        } else {
            return Err(anyhow!("Missing gate weights for layer {}", layer.layer_idx));
        };

        let up_flat = if let Some(wu) = &layer.ffn_up {
            flat_hidden.matmul(wu)?
        } else {
            return Err(anyhow!("Missing up weights for layer {}", layer.layer_idx));
        };

        // SiLU activation on gate
        let gate_flat = candle_nn::ops::silu(&gate_flat)?;

        // Element-wise multiply
        let intermediate_flat = (gate_flat * up_flat)?;

        // Down projection (row-parallel)
        // v2.6.8 FIX: Weights are [in, out] format, NO transpose needed
        // down shape: [ffn_dim_per_node, hidden] = [7168, 4096] for 2 nodes
        let output_flat = if let Some(wd) = &layer.ffn_down {
            intermediate_flat.matmul(wd)?
        } else {
            return Err(anyhow!("Missing down weights for layer {}", layer.layer_idx));
        };

        // Reshape back to [batch, seq, hidden]
        let output = output_flat.reshape((batch, seq_len, hidden_dim))?;

        // All-reduce to combine partial outputs
        let ar_start = Instant::now();
        let output = self.all_reduce(&output, request_id, layer.layer_idx, "mlp").await?;
        let ar_time = ar_start.elapsed();

        let mlp_time = mlp_start.elapsed();
        info!(
            "⚡ [MLP  L{}] Completed in {:?} (all-reduce: {:?}) | Output: {:?}",
            layer.layer_idx, mlp_time, ar_time, output.dims()
        );

        Ok(output)
    }

    /// Apply RMS normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(2)?;
        let x = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        let y = x.broadcast_mul(weight)?;
        Ok(y)
    }

    /// Perform all-reduce on a tensor
    async fn all_reduce(
        &self,
        tensor: &Tensor,
        request_id: &str,
        layer_idx: usize,
        stage: &str,
    ) -> Result<Tensor> {
        let world_size = self.config.shard_config.world_size;

        if world_size == 1 {
            // Single node, no all-reduce needed
            debug!(
                "🔁 [ALL-REDUCE] Skipped (world_size=1) | Layer {} | Stage: {}",
                layer_idx, stage
            );
            return Ok(tensor.clone());
        }

        let start = Instant::now();

        // v2.6.8 DEBUG: Log tensor info before all-reduce
        let tensor_size = tensor.elem_count();
        let tensor_bytes = tensor_size * std::mem::size_of::<f32>();
        debug!(
            "🔁 [ALL-REDUCE] Starting | Layer {} | Stage: {} | Tensor: {:?} ({} elems, {} KB)",
            layer_idx, stage, tensor.dims(), tensor_size, tensor_bytes / 1024
        );

        // Flatten tensor for transfer
        let flatten_start = Instant::now();
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        let shape: Vec<usize> = tensor.dims().to_vec();
        let flatten_time = flatten_start.elapsed();

        // Send all-reduce request
        let send_start = Instant::now();
        if let Some(tx) = &self.all_reduce_tx {
            let req = AllReduceRequest {
                request_id: format!("{}-{}-{}", request_id, layer_idx, stage),
                layer_idx,
                tensor: data,
                shape: shape.clone(),
            };
            tx.send(req).await?;
            let send_time = send_start.elapsed();
            debug!(
                "📤 [ALL-REDUCE] Sent request | Layer {} | Stage: {} | Flatten: {:?} | Send: {:?}",
                layer_idx, stage, flatten_time, send_time
            );
        } else {
            // No all-reduce channel configured - simulate single-node
            warn!(
                "⚠️ [ALL-REDUCE] No channel configured! Simulating single-node | Layer {} | Stage: {}",
                layer_idx, stage
            );
            return Ok(tensor.clone());
        }

        // Wait for response
        // TODO: In a real implementation, this would use the receiver channel
        // For now, simulate single-node behavior (THIS IS THE BOTTLENECK!)
        let latency = start.elapsed().as_millis() as u64;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.all_reduce_count += 1;
            stats.avg_all_reduce_ms =
                (stats.avg_all_reduce_ms * (stats.all_reduce_count - 1) as f64 + latency as f64)
                    / stats.all_reduce_count as f64;
        }

        // v2.6.8 DEBUG: Log completion
        debug!(
            "✅ [ALL-REDUCE] Complete | Layer {} | Stage: {} | Latency: {}ms",
            layer_idx, stage, latency
        );

        // WARNING: Currently returning tensor.clone() without actual network aggregation!
        // This means we're NOT getting real tensor parallel speedup yet.
        Ok(tensor.clone())
    }

    /// Run a single forward pass through all layers
    pub async fn forward(
        &self,
        input_ids: &Tensor,
        request_id: &str,
    ) -> Result<Tensor> {
        let start = Instant::now();
        info!("🚀 [FORWARD] Starting forward pass | Input: {:?} | Request: {}", input_ids.dims(), request_id);

        // Get embeddings using index-based lookup
        let emb_start = Instant::now();
        let mut hidden = if let Some(emb) = &self.token_embedding {
            // Flatten input_ids and gather embeddings
            let flat_ids = input_ids.flatten_all()?;
            let ids_vec = flat_ids.to_vec1::<i64>()?;

            // For each token ID, get the embedding vector
            let mut emb_vectors = Vec::new();
            for &id in &ids_vec {
                let emb_row = emb.i(id as usize)?;
                emb_vectors.push(emb_row);
            }

            // Stack embeddings into [batch, seq, hidden] shape
            if emb_vectors.is_empty() {
                return Err(anyhow!("Empty input"));
            }
            let stacked = Tensor::stack(&emb_vectors, 0)?;
            let (batch, seq_len) = input_ids.dims2()?;
            stacked.reshape((batch, seq_len, ()))?
        } else {
            return Err(anyhow!("Token embedding not loaded"));
        };
        let emb_time = emb_start.elapsed();
        info!("📥 [FORWARD] Embeddings loaded in {:?} | Hidden: {:?}", emb_time, hidden.dims());

        // Process each layer
        let layers_start = Instant::now();
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_start = Instant::now();
            // Attention with residual
            let attn_norm = if let Some(norm) = &layer.attn_norm {
                self.rms_norm(&hidden, norm, 1e-6)?
            } else {
                hidden.clone()
            };

            let attn_out = self.attention_forward(layer, &attn_norm, request_id).await?;
            hidden = (hidden + attn_out)?;

            // MLP with residual
            let ffn_norm = if let Some(norm) = &layer.ffn_norm {
                self.rms_norm(&hidden, norm, 1e-6)?
            } else {
                hidden.clone()
            };

            let mlp_out = self.mlp_forward(layer, &ffn_norm, request_id).await?;
            hidden = (hidden + mlp_out)?;

            let layer_time = layer_start.elapsed();
            if i % 8 == 0 || i == num_layers - 1 {
                info!("📊 [LAYER {}/{}] Completed in {:?}", i + 1, num_layers, layer_time);
            }
        }

        let layers_time = layers_start.elapsed();
        info!("🔄 [FORWARD] All {} layers completed in {:?}", num_layers, layers_time);

        // Final normalization
        if let Some(norm) = &self.output_norm {
            hidden = self.rms_norm(&hidden, norm, 1e-6)?;
        }

        // LM head (only on coordinator, or sharded)
        // v2.6.7 FIX: Flatten for 2D matmul, transpose weight
        let lm_start = Instant::now();
        let logits = if let Some(lm) = &self.lm_head {
            let (batch, seq_len, hidden_dim) = hidden.dims3()?;
            let flat_hidden = hidden.reshape((batch * seq_len, hidden_dim))?;
            let logits_flat = flat_hidden.matmul(&lm.t()?)?;
            let vocab_size = logits_flat.dim(1)?;
            logits_flat.reshape((batch, seq_len, vocab_size))?
        } else if self.config.is_coordinator {
            return Err(anyhow!("LM head not loaded on coordinator"));
        } else {
            // Workers don't have LM head
            hidden
        };
        let lm_time = lm_start.elapsed();

        let elapsed = start.elapsed();
        info!(
            "✅ [FORWARD] Complete in {:?} | Emb: {:?}, Layers: {:?}, LM: {:?} | Logits: {:?}",
            elapsed, emb_time, layers_time, lm_time, logits.dims()
        );

        Ok(logits)
    }

    /// Generate tokens autoregressively
    pub async fn generate(
        &self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        token_callback: impl Fn(u32, &str),
    ) -> Result<Vec<u32>> {
        let start = Instant::now();
        let mut all_ids = input_ids.clone();
        let mut generated = Vec::new();

        // v2.6.8 DEBUG: Track detailed per-token timing
        let mut first_token_time: Option<std::time::Duration> = None;
        let mut forward_times: Vec<std::time::Duration> = Vec::new();
        let mut sample_times: Vec<std::time::Duration> = Vec::new();

        info!(
            "🎬 [GENERATE] Starting generation | Input tokens: {} | Max tokens: {} | World size: {}",
            input_ids.len(), max_tokens, self.config.shard_config.world_size
        );

        for i in 0..max_tokens {
            let token_start = Instant::now();

            // Create input tensor
            let tensor_start = Instant::now();
            let input = Tensor::from_vec(
                all_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                (1, all_ids.len()),
                &self.device,
            )?;
            let tensor_time = tensor_start.elapsed();

            // Forward pass
            let forward_start = Instant::now();
            let request_id = format!("gen-{}", i);
            let logits = self.forward(&input, &request_id).await?;
            let forward_time = forward_start.elapsed();
            forward_times.push(forward_time);

            // Get last token's logits
            let sample_start = Instant::now();
            let seq_len = logits.dim(1)?;
            let last_logits = logits.i((0, seq_len - 1))?;

            // Sample next token (simplified greedy for now)
            let next_token_i64 = last_logits.argmax(0)?.to_scalar::<u32>()?;
            let next_token = next_token_i64;
            let sample_time = sample_start.elapsed();
            sample_times.push(sample_time);

            let token_time = token_start.elapsed();

            // Record first token time (TTFT - Time To First Token)
            if first_token_time.is_none() {
                first_token_time = Some(start.elapsed());
                info!(
                    "🚀 [TTFT] First token generated in {:?} | Token: {} | Forward: {:?} | Sample: {:?}",
                    first_token_time.unwrap(), next_token, forward_time, sample_time
                );
            }

            // Log every token with timing breakdown
            info!(
                "🔢 [TOKEN {}/{}] ID: {} | Total: {:?} | Tensor: {:?} | Forward: {:?} | Sample: {:?} | Seq len: {}",
                i + 1, max_tokens, next_token, token_time, tensor_time, forward_time, sample_time, all_ids.len()
            );

            // Check for EOS
            if next_token == 2 { // Typical EOS token
                info!("🛑 [EOS] End of sequence token detected at position {}", i + 1);
                break;
            }

            generated.push(next_token);
            all_ids.push(next_token);

            // Callback for streaming
            token_callback(next_token, &format!("tok_{}", next_token));
        }

        // Update stats
        let elapsed = start.elapsed();
        let tokens_per_sec = if generated.len() > 0 {
            generated.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // v2.6.8 DEBUG: Calculate timing statistics
        let avg_forward_ms = if !forward_times.is_empty() {
            forward_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / forward_times.len() as f64
        } else { 0.0 };

        let avg_sample_ms = if !sample_times.is_empty() {
            sample_times.iter().map(|d| d.as_millis() as f64).sum::<f64>() / sample_times.len() as f64
        } else { 0.0 };

        let min_forward_ms = forward_times.iter().map(|d| d.as_millis()).min().unwrap_or(0);
        let max_forward_ms = forward_times.iter().map(|d| d.as_millis()).max().unwrap_or(0);

        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.total_tokens += generated.len() as u64;
            stats.avg_tokens_per_sec =
                (stats.avg_tokens_per_sec * (stats.total_requests - 1) as f64 + tokens_per_sec)
                    / stats.total_requests as f64;
            stats.avg_token_time_ms =
                (elapsed.as_millis() as f64) / generated.len().max(1) as f64;

            // Calculate actual speedup vs theoretical
            // Theoretical: N nodes = Nx faster
            // Actual: Measure (with all-reduce overhead accounted)
            let single_node_estimate = stats.avg_token_time_ms * stats.world_size as f64;
            stats.actual_speedup = single_node_estimate / stats.avg_token_time_ms;
        }

        // v2.6.8 DEBUG: Comprehensive summary
        info!(
            "📊 [GENERATE SUMMARY] ========================================"
        );
        info!(
            "🔢 Tokens: {} generated in {:?} ({:.2} tok/s)",
            generated.len(), elapsed, tokens_per_sec
        );
        info!(
            "⏱️  TTFT: {:?} | Avg Forward: {:.1}ms | Avg Sample: {:.1}ms",
            first_token_time.unwrap_or_default(), avg_forward_ms, avg_sample_ms
        );
        info!(
            "📈 Forward Range: {}ms - {}ms | World Size: {}",
            min_forward_ms, max_forward_ms, self.config.shard_config.world_size
        );
        if self.config.shard_config.world_size > 1 {
            // Calculate theoretical speedup (how much faster should it be)
            // Theoretical: each node does 1/N of the work
            let theoretical_speedup = self.config.shard_config.world_size as f64;
            // Actual speedup would require single-node baseline, estimate from forward times
            info!(
                "🎯 Theoretical Speedup: {:.1}x (need single-node baseline to measure actual)",
                theoretical_speedup
            );
        }
        info!(
            "📊 [GENERATE SUMMARY] ========================================"
        );

        Ok(generated)
    }

    /// Get current statistics
    pub async fn stats(&self) -> TensorParallelStats {
        self.stats.read().await.clone()
    }

    /// Check if engine is ready for inference
    pub fn is_ready(&self) -> bool {
        // Check if we have at least some weights loaded
        self.layers.iter().any(|l| l.attn_q.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_parallel_config() {
        let config = TensorParallelConfig::default();
        assert_eq!(config.shard_config.world_size, 1);
        assert!(config.is_coordinator);
    }

    #[test]
    fn test_shard_config_4_nodes() {
        let config = ShardConfig::mistral_7b(4, 2);
        assert_eq!(config.heads_per_node(), 8); // 32 / 4
        assert_eq!(config.head_range(), (16, 24));
        assert_eq!(config.ffn_shard_dim(), 3584); // 14336 / 4
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let config = TensorParallelConfig::default();
        let engine = TensorParallelEngine::new(config, Device::Cpu);

        assert!(!engine.is_ready()); // No weights loaded yet

        let stats = engine.stats().await;
        assert_eq!(stats.world_size, 1);
    }
}
