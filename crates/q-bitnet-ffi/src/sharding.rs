//! Weight Sharding for BitNet Tensor Parallelism
//!
//! This module provides efficient weight sharding for distributed BitNet inference.
//! Ternary weights (2-bit) enable much smaller shards compared to FP16.
//!
//! ## Sharding Strategy
//!
//! For tensor parallelism, we shard along the attention head and FFN dimensions:
//!
//! ```text
//! Layer weights for 4-node tensor parallelism:
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    ATTENTION SHARDING                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Q projection [hidden, hidden]:                              │
//! │    Node 0: heads 0-3     ─┬─ [hidden, hidden/4]             │
//! │    Node 1: heads 4-7     ─┤                                  │
//! │    Node 2: heads 8-11    ─┤  Column-parallel                 │
//! │    Node 3: heads 12-15   ─┘                                  │
//! │                                                              │
//! │  O projection [hidden, hidden]:                              │
//! │    Node 0: partial output ─┬─ [hidden/4, hidden]            │
//! │    Node 1: partial output ─┤                                 │
//! │    Node 2: partial output ─┤  Row-parallel (needs all-reduce)│
//! │    Node 3: partial output ─┘                                 │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      FFN SHARDING                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  Gate/Up [hidden, intermediate]:                             │
//! │    Node 0: dims 0-1407    ─┬─ [hidden, intermediate/4]      │
//! │    Node 1: dims 1408-2815 ─┤                                 │
//! │    Node 2: dims 2816-4223 ─┤  Column-parallel                │
//! │    Node 3: dims 4224-5631 ─┘                                 │
//! │                                                              │
//! │  Down [intermediate, hidden]:                                │
//! │    Node 0: partial output ─┬─ [intermediate/4, hidden]      │
//! │    Node 1: partial output ─┤                                 │
//! │    Node 2: partial output ─┤  Row-parallel (needs all-reduce)│
//! │    Node 3: partial output ─┘                                 │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Savings
//!
//! With 4-node sharding + ternary packing:
//! - Per-node memory: ~100 MB (vs ~2 GB for FP16 unsharded)
//! - Total reduction: ~20x (4x sharding × 5x ternary compression)

use crate::ternary::TernaryTensor;
use crate::engine::{BitNetConfig, BitNetEngine, BitNetLayer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for weight sharding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Total number of nodes
    pub world_size: usize,

    /// This node's rank (0 to world_size-1)
    pub node_rank: usize,

    /// Model configuration
    pub model_config: BitNetConfig,
}

impl ShardConfig {
    /// Create shard config for a specific node
    pub fn new(model_config: BitNetConfig, world_size: usize, node_rank: usize) -> Self {
        assert!(node_rank < world_size, "Node rank must be less than world size");

        Self {
            world_size,
            node_rank,
            model_config,
        }
    }

    /// Number of attention heads per node
    pub fn heads_per_node(&self) -> usize {
        self.model_config.num_heads / self.world_size
    }

    /// Number of KV heads per node (for GQA)
    pub fn kv_heads_per_node(&self) -> usize {
        // KV heads might not divide evenly; handle gracefully
        let total_kv = self.model_config.num_kv_heads;
        if total_kv >= self.world_size {
            total_kv / self.world_size
        } else {
            // If fewer KV heads than nodes, replicate
            total_kv
        }
    }

    /// FFN intermediate dimension per node
    pub fn ffn_dim_per_node(&self) -> usize {
        self.model_config.intermediate_dim / self.world_size
    }

    /// Range of heads for this node
    pub fn head_range(&self) -> (usize, usize) {
        let per_node = self.heads_per_node();
        let start = self.node_rank * per_node;
        let end = start + per_node;
        (start, end)
    }

    /// Range of FFN dimensions for this node
    pub fn ffn_range(&self) -> (usize, usize) {
        let per_node = self.ffn_dim_per_node();
        let start = self.node_rank * per_node;
        let end = start + per_node;
        (start, end)
    }

    /// Memory per node in bytes (ternary packed)
    pub fn memory_per_node(&self) -> usize {
        self.model_config.weight_memory_per_node(self.world_size)
    }
}

/// A sharded BitNet layer for tensor parallelism
#[derive(Debug, Clone)]
pub struct BitNetShardedLayer {
    pub layer_idx: usize,

    /// Sharded attention weights (column-parallel)
    /// Shape: [hidden, heads_per_node × head_dim]
    pub attn_q: Option<TernaryTensor>,
    pub attn_k: Option<TernaryTensor>,
    pub attn_v: Option<TernaryTensor>,

    /// Sharded attention output (row-parallel)
    /// Shape: [heads_per_node × head_dim, hidden]
    pub attn_o: Option<TernaryTensor>,

    /// Sharded FFN gate/up (column-parallel)
    /// Shape: [hidden, ffn_dim_per_node]
    pub ffn_gate: Option<TernaryTensor>,
    pub ffn_up: Option<TernaryTensor>,

    /// Sharded FFN down (row-parallel)
    /// Shape: [ffn_dim_per_node, hidden]
    pub ffn_down: Option<TernaryTensor>,

    /// Normalization weights (replicated, not sharded)
    pub attn_norm: Option<Vec<f32>>,
    pub ffn_norm: Option<Vec<f32>>,
}

impl BitNetShardedLayer {
    /// Create empty sharded layer
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            attn_q: None,
            attn_k: None,
            attn_v: None,
            attn_o: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
            attn_norm: None,
            ffn_norm: None,
        }
    }

    /// Check if layer is complete
    pub fn is_complete(&self) -> bool {
        self.attn_q.is_some() &&
        self.attn_k.is_some() &&
        self.attn_v.is_some() &&
        self.attn_o.is_some() &&
        self.ffn_gate.is_some() &&
        self.ffn_up.is_some() &&
        self.ffn_down.is_some()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(t) = &self.attn_q { total += t.packed.data.len(); }
        if let Some(t) = &self.attn_k { total += t.packed.data.len(); }
        if let Some(t) = &self.attn_v { total += t.packed.data.len(); }
        if let Some(t) = &self.attn_o { total += t.packed.data.len(); }
        if let Some(t) = &self.ffn_gate { total += t.packed.data.len(); }
        if let Some(t) = &self.ffn_up { total += t.packed.data.len(); }
        if let Some(t) = &self.ffn_down { total += t.packed.data.len(); }
        if let Some(n) = &self.attn_norm { total += n.len() * 4; }
        if let Some(n) = &self.ffn_norm { total += n.len() * 4; }
        total
    }
}

/// Sharded BitNet model for tensor parallelism
pub struct BitNetShard {
    /// Shard configuration
    pub config: ShardConfig,

    /// Sharded transformer layers
    pub layers: Vec<BitNetShardedLayer>,

    /// Token embedding (replicated across all nodes)
    pub token_embedding: Option<TernaryTensor>,

    /// Output normalization (replicated)
    pub output_norm: Option<Vec<f32>>,

    /// LM head (can be sharded or replicated)
    pub lm_head: Option<TernaryTensor>,

    /// Whether this node is the coordinator
    pub is_coordinator: bool,
}

impl BitNetShard {
    /// Create new shard
    pub fn new(config: ShardConfig) -> Self {
        let num_layers = config.model_config.num_layers;
        let layers = (0..num_layers)
            .map(|i| BitNetShardedLayer::new(i))
            .collect();

        Self {
            config,
            layers,
            token_embedding: None,
            output_norm: None,
            lm_head: None,
            is_coordinator: false,
        }
    }

    /// Create shard from full model
    ///
    /// Extracts this node's portion of the weights.
    pub fn from_full_model(engine: &BitNetEngine, config: ShardConfig) -> Result<Self> {
        info!(
            "📦 Creating shard {}/{} from full model...",
            config.node_rank,
            config.world_size
        );

        let mut shard = Self::new(config.clone());

        // Shard each layer
        for (i, layer) in engine.layers.iter().enumerate() {
            shard.layers[i] = shard.shard_layer(layer, &config)?;
        }

        // Replicate embedding and output
        shard.token_embedding = engine.token_embedding.clone();
        shard.output_norm = engine.output_norm.clone();
        shard.lm_head = engine.lm_head.clone();

        let memory = shard.memory_bytes();
        info!(
            "✅ Shard {}/{} created: {} MB ({}x reduction)",
            config.node_rank,
            config.world_size,
            memory / (1024 * 1024),
            engine.memory_usage() / memory.max(1)
        );

        Ok(shard)
    }

    /// Shard a single layer
    fn shard_layer(&self, layer: &BitNetLayer, config: &ShardConfig) -> Result<BitNetShardedLayer> {
        let mut sharded = BitNetShardedLayer::new(layer.layer_idx);

        let hidden = config.model_config.hidden_dim;
        let heads_per_node = config.heads_per_node();
        let head_dim = config.model_config.head_dim;
        let (head_start, head_end) = config.head_range();
        let (ffn_start, ffn_end) = config.ffn_range();

        // Shard Q (column-parallel by head)
        if let Some(q) = &layer.attn_q {
            // Original shape: [hidden, hidden]
            // Sharded shape: [hidden, heads_per_node × head_dim]
            let shard_start = head_start * head_dim;
            let shard_end = head_end * head_dim;

            // Extract columns for this node's heads
            sharded.attn_q = Some(self.shard_columns(q, hidden, shard_start, shard_end)?);
        }

        // Shard K, V (similar to Q but with KV heads)
        let kv_heads_per_node = config.kv_heads_per_node();
        let kv_shard_start = (config.node_rank * kv_heads_per_node) * head_dim;
        let kv_shard_end = kv_shard_start + kv_heads_per_node * head_dim;

        if let Some(k) = &layer.attn_k {
            sharded.attn_k = Some(self.shard_columns(k, hidden, kv_shard_start, kv_shard_end)?);
        }
        if let Some(v) = &layer.attn_v {
            sharded.attn_v = Some(self.shard_columns(v, hidden, kv_shard_start, kv_shard_end)?);
        }

        // Shard O (row-parallel: each node has partial input)
        if let Some(o) = &layer.attn_o {
            let shard_start = head_start * head_dim;
            let shard_end = head_end * head_dim;
            sharded.attn_o = Some(self.shard_rows(o, shard_start, shard_end, hidden)?);
        }

        // Shard FFN gate/up (column-parallel)
        if let Some(gate) = &layer.ffn_gate {
            sharded.ffn_gate = Some(self.shard_columns(gate, hidden, ffn_start, ffn_end)?);
        }
        if let Some(up) = &layer.ffn_up {
            sharded.ffn_up = Some(self.shard_columns(up, hidden, ffn_start, ffn_end)?);
        }

        // Shard FFN down (row-parallel)
        if let Some(down) = &layer.ffn_down {
            sharded.ffn_down = Some(self.shard_rows(down, ffn_start, ffn_end, hidden)?);
        }

        // Copy norms (replicated)
        sharded.attn_norm = layer.attn_norm.clone();
        sharded.ffn_norm = layer.ffn_norm.clone();

        debug!(
            "📊 Layer {} shard: {} bytes",
            layer.layer_idx,
            sharded.memory_bytes()
        );

        Ok(sharded)
    }

    /// Shard columns of a weight matrix
    ///
    /// For column-parallel: [in_dim, out_dim] -> [in_dim, shard_out_dim]
    fn shard_columns(
        &self,
        tensor: &TernaryTensor,
        in_dim: usize,
        col_start: usize,
        col_end: usize,
    ) -> Result<TernaryTensor> {
        let shard_cols = col_end - col_start;
        let mut values = Vec::with_capacity(in_dim * shard_cols);

        for row in 0..in_dim {
            for col in col_start..col_end {
                let idx = row * tensor.shape()[1] + col;
                if let Some(v) = tensor.packed.get(idx) {
                    values.push(v);
                } else {
                    return Err(anyhow!("Index out of bounds in column sharding"));
                }
            }
        }

        Ok(TernaryTensor {
            packed: crate::ternary::PackedTernary::from_values(&values),
            shape: vec![in_dim, shard_cols],
            scale: tensor.scale,
        })
    }

    /// Shard rows of a weight matrix
    ///
    /// For row-parallel: [in_dim, out_dim] -> [shard_in_dim, out_dim]
    fn shard_rows(
        &self,
        tensor: &TernaryTensor,
        row_start: usize,
        row_end: usize,
        out_dim: usize,
    ) -> Result<TernaryTensor> {
        let shard_rows = row_end - row_start;
        let mut values = Vec::with_capacity(shard_rows * out_dim);

        for row in row_start..row_end {
            for col in 0..out_dim {
                let idx = row * out_dim + col;
                if let Some(v) = tensor.packed.get(idx) {
                    values.push(v);
                } else {
                    return Err(anyhow!("Index out of bounds in row sharding"));
                }
            }
        }

        Ok(TernaryTensor {
            packed: crate::ternary::PackedTernary::from_values(&values),
            shape: vec![shard_rows, out_dim],
            scale: tensor.scale,
        })
    }

    /// Total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;

        for layer in &self.layers {
            total += layer.memory_bytes();
        }

        if let Some(t) = &self.token_embedding {
            total += t.packed.data.len();
        }
        if let Some(t) = &self.lm_head {
            total += t.packed.data.len();
        }
        if let Some(n) = &self.output_norm {
            total += n.len() * 4;
        }

        total
    }

    /// Check if shard is ready for inference
    pub fn is_ready(&self) -> bool {
        self.token_embedding.is_some() &&
        self.lm_head.is_some() &&
        self.layers.iter().all(|l| l.is_complete())
    }
}

/// Coordinator for distributed BitNet sharding
///
/// Manages weight distribution and shard creation across nodes.
pub struct ShardCoordinator {
    /// Full model (coordinator only)
    full_model: Option<Arc<BitNetEngine>>,

    /// World size
    world_size: usize,

    /// This node's rank
    node_rank: usize,
}

impl ShardCoordinator {
    /// Create new coordinator
    pub fn new(world_size: usize, node_rank: usize) -> Self {
        Self {
            full_model: None,
            world_size,
            node_rank,
        }
    }

    /// Load full model (coordinator only)
    pub fn load_model(&mut self, model: BitNetEngine) {
        self.full_model = Some(Arc::new(model));
    }

    /// Create shard for a specific node
    pub fn create_shard(&self, target_rank: usize) -> Result<BitNetShard> {
        let model = self.full_model.as_ref()
            .ok_or_else(|| anyhow!("Full model not loaded"))?;

        let config = ShardConfig::new(
            model.config.clone(),
            self.world_size,
            target_rank,
        );

        BitNetShard::from_full_model(model, config)
    }

    /// Serialize shard for network transfer
    pub fn serialize_shard(&self, shard: &BitNetShard) -> Result<Vec<u8>> {
        // Ternary packing means this is 16x smaller than FP16!
        let mut data = Vec::new();

        // Serialize config
        let config_bytes = bincode::serialize(&shard.config)?;
        data.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&config_bytes);

        // Serialize layers
        for layer in &shard.layers {
            // Each ternary tensor is already packed
            if let Some(t) = &layer.attn_q {
                data.extend_from_slice(&(t.packed.data.len() as u32).to_le_bytes());
                data.extend_from_slice(&t.packed.data);
            }
            // ... serialize other tensors
        }

        Ok(data)
    }

    /// Deserialize shard from network transfer
    pub fn deserialize_shard(&self, _data: &[u8]) -> Result<BitNetShard> {
        // TODO: Implement deserialization
        Err(anyhow!("Not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_config() {
        let model_config = BitNetConfig::default();
        let config = ShardConfig::new(model_config, 4, 2);

        assert_eq!(config.world_size, 4);
        assert_eq!(config.node_rank, 2);
        assert_eq!(config.heads_per_node(), 4); // 16 heads / 4 nodes

        let (start, end) = config.head_range();
        assert_eq!(start, 8); // Node 2 gets heads 8-11
        assert_eq!(end, 12);
    }

    #[test]
    fn test_ffn_sharding() {
        let model_config = BitNetConfig::default();
        let config = ShardConfig::new(model_config, 4, 0);

        // Intermediate dim is 5632
        assert_eq!(config.ffn_dim_per_node(), 1408);

        let (start, end) = config.ffn_range();
        assert_eq!(start, 0);
        assert_eq!(end, 1408);
    }

    #[test]
    fn test_memory_per_node() {
        let model_config = BitNetConfig::default();

        // Single node
        let single = model_config.weight_memory_bytes();

        // 4-node sharding
        let per_node = model_config.weight_memory_per_node(4);

        // Should be roughly 1/4 (plus replicated weights)
        assert!(per_node < single);
        assert!(per_node > single / 8); // Not exactly 1/4 due to replicated embedding

        println!("Single node: {} MB", single / (1024 * 1024));
        println!("Per node (4-way): {} MB", per_node / (1024 * 1024));
    }
}
