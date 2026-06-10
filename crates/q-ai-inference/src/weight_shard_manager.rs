//! Weight Shard Manager for Tensor Parallelism
//! v2.4.1-beta: TemporalShield-protected weight updates (3-of-5 threshold, HNDL-resistant)
//!
//! Implements dimension-based weight sharding for distributed tensor parallelism.
//! Unlike pipeline parallelism (which splits by layers), tensor parallelism splits
//! each layer's weights across multiple nodes for true parallel speedup.
//!
//! ## Security (v2.4.1+)
//!
//! Weight updates are protected with TemporalShield to prevent:
//! - Model IP theft (proprietary weights encrypted with threshold shares)
//! - Gradient poisoning (STARK proofs verify update validity)
//! - HNDL attacks (ML-KEM-1024 post-quantum encryption)
//!
//! ## Architecture
//!
//! ```text
//! Traditional (Single Node):
//! ┌─────────────────────────────────────────────────┐
//! │ Full Layer (4096 hidden dims)                   │
//! │ MatMul: [seq, 4096] × [4096, 4096] = 4096² ops  │
//! │ Time: 100ms                                      │
//! └─────────────────────────────────────────────────┘
//!
//! Tensor Parallel (4 Nodes):
//! ┌────────────┬────────────┬────────────┬────────────┐
//! │ Node A     │ Node B     │ Node C     │ Node D     │
//! │ dims 0-1023│ 1024-2047  │ 2048-3071  │ 3072-4095  │
//! │ 1024² ops  │ 1024² ops  │ 1024² ops  │ 1024² ops  │
//! │ Time: 25ms │ Time: 25ms │ Time: 25ms │ Time: 25ms │
//! └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
//!       └────────────┴────────────┴────────────┘
//!                         │
//!               All-Reduce (gather)
//!                         │
//!                   Total: ~30ms (4x faster!)
//! ```
//!
//! ## Sharding Strategy
//!
//! For Mistral-7B (hidden_dim=4096, num_heads=32):
//!
//! **Attention (Column Parallel):**
//! - Q projection: [4096, 4096] → N shards of [4096, 4096/N]
//! - K projection: [4096, 1024] → N shards of [4096, 1024/N]
//! - V projection: [4096, 1024] → N shards of [4096, 1024/N]
//! - Each node handles `32/N` attention heads
//!
//! **MLP (Column + Row Parallel):**
//! - Gate: [4096, 14336] → N shards of [4096, 14336/N]
//! - Up:   [4096, 14336] → N shards of [4096, 14336/N]
//! - Down: [14336, 4096] → N shards of [14336/N, 4096]

use anyhow::{anyhow, Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// TemporalShield threshold parameters for weight updates (3-of-5)
pub const TEMPORAL_WEIGHT_THRESHOLD: usize = 3;
pub const TEMPORAL_WEIGHT_TOTAL_TRUSTEES: usize = 5;

/// v2.4.1-beta: TemporalShield-protected weight/gradient update
///
/// Protects distributed AI tensor updates with (3,5) threshold secret sharing.
/// Prevents model IP theft, gradient poisoning, and HNDL attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWeightUpdate {
    /// Layer index this update applies to
    pub layer_idx: usize,
    /// Weight name (e.g., "attn_q", "ffn_gate")
    pub weight_name: String,
    /// TemporalEnvelope containing protected gradient data
    /// Encoded bytes from TemporalEnvelope::to_bytes()
    pub protected_gradient: Vec<u8>,
    /// Training round when this update was created
    pub reveal_round: u64,
    /// STARK proof of gradient validity (NO TRUSTED SETUP)
    /// Proves gradient was computed correctly without revealing weights
    pub validity_proof: Vec<u8>,
    /// Gradient checksum for quick validation
    pub gradient_checksum: [u8; 32],
    /// Node that created this update
    pub source_node: String,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Number of trustees who have provided decryption shares
    pub shares_available: usize,
    /// Whether gradient can be applied (shares_available >= threshold)
    pub can_apply: bool,
    /// Gradient magnitude (L2 norm) for anomaly detection
    pub gradient_magnitude: f32,
    /// Key commitment for verification
    pub key_commitment: [u8; 32],
}

impl TemporalWeightUpdate {
    /// Create a new temporal weight update
    pub fn new(
        layer_idx: usize,
        weight_name: String,
        protected_gradient: Vec<u8>,
        reveal_round: u64,
        validity_proof: Vec<u8>,
        gradient_checksum: [u8; 32],
        source_node: String,
        gradient_magnitude: f32,
        key_commitment: [u8; 32],
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            layer_idx,
            weight_name,
            protected_gradient,
            reveal_round,
            validity_proof,
            gradient_checksum,
            source_node,
            created_at: timestamp,
            shares_available: 0,
            can_apply: false,
            gradient_magnitude,
            key_commitment,
        }
    }

    /// Record a decryption share and update application status
    pub fn record_share(&mut self) {
        self.shares_available += 1;
        if self.shares_available >= TEMPORAL_WEIGHT_THRESHOLD {
            self.can_apply = true;
        }
    }

    /// Check if gradient magnitude is within acceptable bounds
    /// (Simple anomaly detection for gradient poisoning)
    pub fn is_magnitude_valid(&self, max_magnitude: f32) -> bool {
        self.gradient_magnitude > 0.0 && self.gradient_magnitude <= max_magnitude
    }

    /// Unique identifier for this update
    pub fn update_id(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.layer_idx,
            self.weight_name,
            self.reveal_round,
            hex::encode(&self.gradient_checksum[..8])
        )
    }

    /// Serialize to bytes for storage/transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes)
            .map_err(|e| format!("Deserialization failed: {}", e))
    }
}

/// Status of a temporal weight update
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemporalUpdateStatus {
    /// Update protected, waiting for shares
    Protected,
    /// Sufficient shares available, can apply
    Ready,
    /// Update has been applied to weights
    Applied,
    /// Update rejected (invalid proof or gradient poisoning detected)
    Rejected,
    /// Update expired (not enough shares in time)
    Expired,
}

/// Statistics for temporal weight update system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWeightStats {
    /// Total updates created
    pub total_updates: usize,
    /// Updates currently protected
    pub protected_updates: usize,
    /// Updates ready for application
    pub ready_updates: usize,
    /// Updates successfully applied
    pub applied_updates: usize,
    /// Updates rejected (poisoning/invalid)
    pub rejected_updates: usize,
    /// Average shares per update
    pub avg_shares_per_update: f32,
    /// Average gradient magnitude
    pub avg_gradient_magnitude: f32,
}

/// Registry entry for tracking temporal weight updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalUpdateEntry {
    /// Unique update ID
    pub update_id: String,
    /// Layer and weight name
    pub layer_idx: usize,
    pub weight_name: String,
    /// Training round
    pub reveal_round: u64,
    /// Current status
    pub status: TemporalUpdateStatus,
    /// Gradient checksum
    pub gradient_checksum: [u8; 32],
    /// Source node
    pub source_node: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Application timestamp (0 if not applied)
    pub applied_at: u64,
}

/// Configuration for weight sharding across nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Total number of nodes in the tensor parallel group
    pub world_size: usize,

    /// This node's rank (0 to world_size-1)
    pub node_rank: usize,

    /// Model hidden dimension (4096 for Mistral-7B)
    pub hidden_dim: usize,

    /// Number of attention heads (32 for Mistral-7B)
    pub num_attention_heads: usize,

    /// Number of key-value heads (8 for Mistral-7B GQA)
    pub num_kv_heads: usize,

    /// FFN intermediate dimension (14336 for Mistral-7B)
    pub ffn_dim: usize,

    /// Number of transformer layers (32 for Mistral-7B)
    pub num_layers: usize,

    /// Head dimension (128 for Mistral-7B)
    pub head_dim: usize,
}

impl ShardConfig {
    /// Create config for Mistral-7B
    pub fn mistral_7b(world_size: usize, node_rank: usize) -> Self {
        Self {
            world_size,
            node_rank,
            hidden_dim: 4096,
            num_attention_heads: 32,
            num_kv_heads: 8,
            ffn_dim: 14336,
            num_layers: 32,
            head_dim: 128,
        }
    }

    /// Create config for Ministral-3B (smaller model for testing)
    pub fn ministral_3b(world_size: usize, node_rank: usize) -> Self {
        Self {
            world_size,
            node_rank,
            hidden_dim: 2560,
            num_attention_heads: 32,
            num_kv_heads: 8,
            ffn_dim: 8960,
            num_layers: 24,
            head_dim: 80,
        }
    }

    /// Number of attention heads per node
    pub fn heads_per_node(&self) -> usize {
        self.num_attention_heads / self.world_size
    }

    /// Number of KV heads per node
    pub fn kv_heads_per_node(&self) -> usize {
        // GQA: fewer KV heads, shared across query heads
        (self.num_kv_heads / self.world_size).max(1)
    }

    /// Q/K/V output dimension per node
    pub fn attn_shard_dim(&self) -> usize {
        self.heads_per_node() * self.head_dim
    }

    /// FFN intermediate dimension per node
    pub fn ffn_shard_dim(&self) -> usize {
        self.ffn_dim / self.world_size
    }

    /// Range of attention heads for this node
    pub fn head_range(&self) -> (usize, usize) {
        let heads_per = self.heads_per_node();
        let start = self.node_rank * heads_per;
        let end = start + heads_per;
        (start, end)
    }

    /// Range of FFN dimensions for this node
    pub fn ffn_range(&self) -> (usize, usize) {
        let dim_per = self.ffn_shard_dim();
        let start = self.node_rank * dim_per;
        let end = start + dim_per;
        (start, end)
    }
}

/// A sharded weight tensor for a single layer component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightShard {
    /// Layer index this weight belongs to
    pub layer_idx: usize,

    /// Weight name (e.g., "attn_q", "ffn_gate")
    pub weight_name: String,

    /// Sharded tensor data (serialized f32)
    pub data: Vec<u8>,

    /// Shape of the sharded tensor
    pub shape: Vec<usize>,

    /// Which dimension was sharded (0 or 1)
    pub shard_dim: usize,

    /// Original full shape before sharding
    pub original_shape: Vec<usize>,

    /// Is the data compressed?
    pub compressed: bool,
}

impl WeightShard {
    /// Create a new weight shard from f32 data
    pub fn new(
        layer_idx: usize,
        weight_name: String,
        data: Vec<f32>,
        shape: Vec<usize>,
        shard_dim: usize,
        original_shape: Vec<usize>,
        compress: bool,
    ) -> Self {
        let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let (final_data, compressed) = if compress {
            (compress_data(&raw_bytes), true)
        } else {
            (raw_bytes, false)
        };

        Self {
            layer_idx,
            weight_name,
            data: final_data,
            shape,
            shard_dim,
            original_shape,
            compressed,
        }
    }

    /// Convert to f32 vector
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        let bytes = if self.compressed {
            decompress_data(&self.data)?
        } else {
            self.data.clone()
        };

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    /// Convert to Candle Tensor
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let data = self.to_f32_vec()?;
        let shape: Vec<usize> = self.shape.clone();
        let tensor = Tensor::from_vec(data, shape.as_slice(), device)?;
        Ok(tensor)
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Manages weight sharding for tensor parallelism
pub struct WeightShardManager {
    /// Shard configuration
    pub config: ShardConfig,

    /// Path to the GGUF model file
    model_path: PathBuf,

    /// Device for tensor operations
    device: Device,

    /// Cached sharded weights (layer_idx, weight_name) -> WeightShard
    shards: Arc<RwLock<HashMap<(usize, String), WeightShard>>>,

    /// Whether to compress shards for P2P transfer
    compression_enabled: bool,
}

impl WeightShardManager {
    /// Create a new weight shard manager
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: ShardConfig,
        device: Device,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", model_path.display()));
        }

        info!(
            "🔧 WeightShardManager initialized: rank={}/{}, model={}",
            config.node_rank,
            config.world_size,
            model_path.display()
        );

        Ok(Self {
            config,
            model_path,
            device,
            shards: Arc::new(RwLock::new(HashMap::new())),
            compression_enabled: true,
        })
    }

    /// Load and shard weights for all layers (coordinator only)
    ///
    /// This loads the full model and creates shards for all nodes.
    /// Called by the coordinator, which then distributes shards via P2P.
    pub async fn load_and_shard_all(&self) -> Result<HashMap<usize, Vec<WeightShard>>> {
        info!(
            "📦 Loading and sharding model for {} nodes...",
            self.config.world_size
        );

        let mut file = File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF file")?;

        info!("✅ GGUF loaded: {} tensors", content.tensor_infos.len());

        // Map from node_rank -> Vec<WeightShard>
        let mut node_shards: HashMap<usize, Vec<WeightShard>> = HashMap::new();
        for rank in 0..self.config.world_size {
            node_shards.insert(rank, Vec::new());
        }

        // ⚡ v2.6.3: Load embedding and output layers FIRST (required for inference!)
        self.load_embedding_and_output_layers(
            &mut file,
            &content,
            &mut node_shards,
        )?;

        // Shard each layer's weights
        for layer_idx in 0..self.config.num_layers {
            debug!("Sharding layer {}", layer_idx);

            // Attention weights - Column parallel (shard output dim)
            self.shard_attention_weights(
                &mut file,
                &content,
                layer_idx,
                &mut node_shards,
            )?;

            // FFN weights - Column parallel for gate/up, Row parallel for down
            self.shard_ffn_weights(
                &mut file,
                &content,
                layer_idx,
                &mut node_shards,
            )?;

            // Normalization weights are replicated (small, not worth sharding)
            self.replicate_norm_weights(
                &mut file,
                &content,
                layer_idx,
                &mut node_shards,
            )?;
        }

        let total_shards: usize = node_shards.values().map(|v| v.len()).sum();
        info!(
            "✅ Created {} shards across {} nodes",
            total_shards,
            self.config.world_size
        );

        Ok(node_shards)
    }

    /// Shard attention weights (Q, K, V, O) for a single layer
    fn shard_attention_weights(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        layer_idx: usize,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        let prefix = format!("blk.{}", layer_idx);

        // Q projection: [hidden_dim, hidden_dim] -> Column parallel
        if let Some(q_tensor) = self.try_load_qtensor(file, content, &format!("{}.attn_q.weight", prefix))? {
            let q_data = dequantize_tensor(&q_tensor)?;
            let shape = q_tensor.shape().dims().to_vec();

            self.column_shard_weight(
                layer_idx,
                "attn_q".to_string(),
                q_data,
                shape,
                node_shards,
            )?;
        }

        // K projection: GGUF format [kv_dim, hidden_dim] = [1024, 4096] for Mistral GQA
        // v2.6.9 FIX: Use ROW sharding (split dim 0) since output is kv_dim
        // Shard shape: [512, 4096] per node (with 2 nodes)
        if let Some(k_tensor) = self.try_load_qtensor(file, content, &format!("{}.attn_k.weight", prefix))? {
            let k_data = dequantize_tensor(&k_tensor)?;
            let shape = k_tensor.shape().dims().to_vec();

            // For GQA with unequal dims, use row sharding on output dimension
            self.row_shard_weight(
                layer_idx,
                "attn_k".to_string(),
                k_data,
                shape,
                node_shards,
            )?;
        }

        // V projection: GGUF format [kv_dim, hidden_dim] = [1024, 4096] for Mistral GQA
        // v2.6.9 FIX: Use ROW sharding (split dim 0) since output is kv_dim
        if let Some(v_tensor) = self.try_load_qtensor(file, content, &format!("{}.attn_v.weight", prefix))? {
            let v_data = dequantize_tensor(&v_tensor)?;
            let shape = v_tensor.shape().dims().to_vec();

            self.row_shard_weight(
                layer_idx,
                "attn_v".to_string(),
                v_data,
                shape,
                node_shards,
            )?;
        }

        // Output projection: [hidden_dim, hidden_dim] -> Row parallel
        if let Some(o_tensor) = self.try_load_qtensor(file, content, &format!("{}.attn_output.weight", prefix))? {
            let o_data = dequantize_tensor(&o_tensor)?;
            let shape = o_tensor.shape().dims().to_vec();

            self.row_shard_weight(
                layer_idx,
                "attn_output".to_string(),
                o_data,
                shape,
                node_shards,
            )?;
        }

        Ok(())
    }

    /// Shard FFN weights (gate, up, down) for a single layer
    fn shard_ffn_weights(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        layer_idx: usize,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        let prefix = format!("blk.{}", layer_idx);

        // Gate projection: [hidden_dim, ffn_dim] -> Column parallel
        if let Some(gate_tensor) = self.try_load_qtensor(file, content, &format!("{}.ffn_gate.weight", prefix))? {
            let gate_data = dequantize_tensor(&gate_tensor)?;
            let shape = gate_tensor.shape().dims().to_vec();

            self.column_shard_weight(
                layer_idx,
                "ffn_gate".to_string(),
                gate_data,
                shape,
                node_shards,
            )?;
        }

        // Up projection: [hidden_dim, ffn_dim] -> Column parallel
        if let Some(up_tensor) = self.try_load_qtensor(file, content, &format!("{}.ffn_up.weight", prefix))? {
            let up_data = dequantize_tensor(&up_tensor)?;
            let shape = up_tensor.shape().dims().to_vec();

            self.column_shard_weight(
                layer_idx,
                "ffn_up".to_string(),
                up_data,
                shape,
                node_shards,
            )?;
        }

        // Down projection: [ffn_dim, hidden_dim] -> Row parallel
        if let Some(down_tensor) = self.try_load_qtensor(file, content, &format!("{}.ffn_down.weight", prefix))? {
            let down_data = dequantize_tensor(&down_tensor)?;
            let shape = down_tensor.shape().dims().to_vec();

            self.row_shard_weight(
                layer_idx,
                "ffn_down".to_string(),
                down_data,
                shape,
                node_shards,
            )?;
        }

        Ok(())
    }

    /// Load embedding and output layers (required for inference!)
    /// These must be replicated to ALL nodes because:
    /// - Token embedding: discrete lookup, each token needs full embedding vector
    /// - Final norm: 1D, needs full hidden_dim
    /// - LM head: maps hidden_dim -> vocab_size for token prediction
    fn load_embedding_and_output_layers(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        info!("📥 Loading embedding and output layers (replicated to all nodes)...");

        // Token embedding - try multiple naming conventions
        let embedding_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "wte.weight",
        ];

        let mut found_embedding = false;
        for name in &embedding_names {
            if let Some(emb_tensor) = self.try_load_qtensor(file, content, name)? {
                let data = dequantize_tensor(&emb_tensor)?;
                let shape = emb_tensor.shape().dims().to_vec();

                info!("✅ Loaded token embedding '{}': {:?}", name, shape);

                // Replicate to ALL nodes (use layer_idx = usize::MAX to indicate global)
                for rank in 0..self.config.world_size {
                    let shard = WeightShard::new(
                        usize::MAX, // Special marker for embedding layer
                        "token_embd".to_string(),
                        data.clone(),
                        shape.clone(),
                        0, // Not sharded
                        shape.clone(),
                        self.compression_enabled,
                    );
                    node_shards.get_mut(&rank).unwrap().push(shard);
                }
                found_embedding = true;
                break;
            }
        }

        if !found_embedding {
            warn!("⚠️ Token embedding not found in model! Tried: {:?}", embedding_names);
            // List available tensors for debugging
            let available: Vec<_> = content.tensor_infos.keys().take(20).collect();
            debug!("Available tensors (first 20): {:?}", available);
        }

        // Final layer norm - try multiple naming conventions
        let norm_names = [
            "output_norm.weight",
            "norm.weight",
            "model.norm.weight",
            "ln_f.weight",
        ];

        let mut found_norm = false;
        for name in &norm_names {
            if let Some(norm_tensor) = self.try_load_qtensor(file, content, name)? {
                let data = dequantize_tensor(&norm_tensor)?;
                let shape = norm_tensor.shape().dims().to_vec();

                info!("✅ Loaded final norm '{}': {:?}", name, shape);

                for rank in 0..self.config.world_size {
                    let shard = WeightShard::new(
                        usize::MAX,
                        "output_norm".to_string(),
                        data.clone(),
                        shape.clone(),
                        0,
                        shape.clone(),
                        self.compression_enabled,
                    );
                    node_shards.get_mut(&rank).unwrap().push(shard);
                }
                found_norm = true;
                break;
            }
        }

        if !found_norm {
            warn!("⚠️ Final norm not found in model! Tried: {:?}", norm_names);
        }

        // LM head (output projection) - try multiple naming conventions
        let output_names = [
            "output.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ];

        let mut found_output = false;
        for name in &output_names {
            if let Some(out_tensor) = self.try_load_qtensor(file, content, name)? {
                let data = dequantize_tensor(&out_tensor)?;
                let shape = out_tensor.shape().dims().to_vec();

                info!("✅ Loaded LM head '{}': {:?}", name, shape);

                for rank in 0..self.config.world_size {
                    let shard = WeightShard::new(
                        usize::MAX,
                        "output".to_string(),
                        data.clone(),
                        shape.clone(),
                        0,
                        shape.clone(),
                        self.compression_enabled,
                    );
                    node_shards.get_mut(&rank).unwrap().push(shard);
                }
                found_output = true;
                break;
            }
        }

        if !found_output {
            warn!("⚠️ LM head not found in model! Tried: {:?}", output_names);
        }

        // Summary
        let loaded_count = (found_embedding as u8) + (found_norm as u8) + (found_output as u8);
        info!("📦 Loaded {}/3 embedding/output layers", loaded_count);

        if loaded_count == 0 {
            return Err(anyhow!("Failed to load any embedding/output layers - model may be incompatible"));
        }

        Ok(())
    }

    /// Replicate normalization weights to all nodes (small, not worth sharding)
    fn replicate_norm_weights(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        layer_idx: usize,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm
        if let Some(attn_norm) = self.try_load_qtensor(file, content, &format!("{}.attn_norm.weight", prefix))? {
            let data = dequantize_tensor(&attn_norm)?;
            let shape = attn_norm.shape().dims().to_vec();

            // Replicate to all nodes
            for rank in 0..self.config.world_size {
                let shard = WeightShard::new(
                    layer_idx,
                    "attn_norm".to_string(),
                    data.clone(),
                    shape.clone(),
                    0, // Not sharded
                    shape.clone(),
                    self.compression_enabled,
                );
                node_shards.get_mut(&rank).unwrap().push(shard);
            }
        }

        // FFN norm
        if let Some(ffn_norm) = self.try_load_qtensor(file, content, &format!("{}.ffn_norm.weight", prefix))? {
            let data = dequantize_tensor(&ffn_norm)?;
            let shape = ffn_norm.shape().dims().to_vec();

            for rank in 0..self.config.world_size {
                let shard = WeightShard::new(
                    layer_idx,
                    "ffn_norm".to_string(),
                    data.clone(),
                    shape.clone(),
                    0,
                    shape.clone(),
                    self.compression_enabled,
                );
                node_shards.get_mut(&rank).unwrap().push(shard);
            }
        }

        Ok(())
    }

    /// Column-parallel sharding: split output dimension
    /// Used for Q/K/V/gate/up projections
    fn column_shard_weight(
        &self,
        layer_idx: usize,
        weight_name: String,
        data: Vec<f32>,
        shape: Vec<usize>,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        if shape.len() != 2 {
            return Err(anyhow!("Expected 2D tensor for column sharding"));
        }

        let (in_dim, out_dim) = (shape[0], shape[1]);
        let shard_size = out_dim / self.config.world_size;

        if out_dim % self.config.world_size != 0 {
            warn!(
                "Output dim {} not divisible by world_size {}, padding may be needed",
                out_dim,
                self.config.world_size
            );
        }

        for rank in 0..self.config.world_size {
            let start_col = rank * shard_size;
            let end_col = ((rank + 1) * shard_size).min(out_dim);
            let actual_shard_size = end_col - start_col;

            // Extract columns for this rank
            let mut shard_data = Vec::with_capacity(in_dim * actual_shard_size);
            for row in 0..in_dim {
                for col in start_col..end_col {
                    shard_data.push(data[row * out_dim + col]);
                }
            }

            let shard = WeightShard::new(
                layer_idx,
                weight_name.clone(),
                shard_data,
                vec![in_dim, actual_shard_size],
                1, // Sharded on dim 1 (columns)
                shape.clone(),
                self.compression_enabled,
            );

            node_shards.get_mut(&rank).unwrap().push(shard);
        }

        Ok(())
    }

    /// Row-parallel sharding: split input dimension
    /// Used for output/down projections
    fn row_shard_weight(
        &self,
        layer_idx: usize,
        weight_name: String,
        data: Vec<f32>,
        shape: Vec<usize>,
        node_shards: &mut HashMap<usize, Vec<WeightShard>>,
    ) -> Result<()> {
        if shape.len() != 2 {
            return Err(anyhow!("Expected 2D tensor for row sharding"));
        }

        let (in_dim, out_dim) = (shape[0], shape[1]);
        let shard_size = in_dim / self.config.world_size;

        for rank in 0..self.config.world_size {
            let start_row = rank * shard_size;
            let end_row = ((rank + 1) * shard_size).min(in_dim);
            let actual_shard_size = end_row - start_row;

            // Extract rows for this rank
            let mut shard_data = Vec::with_capacity(actual_shard_size * out_dim);
            for row in start_row..end_row {
                for col in 0..out_dim {
                    shard_data.push(data[row * out_dim + col]);
                }
            }

            let shard = WeightShard::new(
                layer_idx,
                weight_name.clone(),
                shard_data,
                vec![actual_shard_size, out_dim],
                0, // Sharded on dim 0 (rows)
                shape.clone(),
                self.compression_enabled,
            );

            node_shards.get_mut(&rank).unwrap().push(shard);
        }

        Ok(())
    }

    /// Try to load a quantized tensor by name
    fn try_load_qtensor(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        name: &str,
    ) -> Result<Option<QTensor>> {
        if let Some(tensor_info) = content.tensor_infos.get(name) {
            let qtensor = tensor_info
                .read(file, content.tensor_data_offset, &self.device)
                .with_context(|| format!("Failed to read tensor '{}'", name))?;
            Ok(Some(qtensor))
        } else {
            Ok(None)
        }
    }

    /// Store received shards from coordinator
    pub async fn store_shards(&self, shards: Vec<WeightShard>) {
        let mut cache = self.shards.write().await;
        for shard in shards {
            let key = (shard.layer_idx, shard.weight_name.clone());
            cache.insert(key, shard);
        }
    }

    /// Get a cached shard
    pub async fn get_shard(&self, layer_idx: usize, weight_name: &str) -> Option<WeightShard> {
        self.shards.read().await.get(&(layer_idx, weight_name.to_string())).cloned()
    }

    /// Get all shards for a layer
    pub async fn get_layer_shards(&self, layer_idx: usize) -> Vec<WeightShard> {
        self.shards
            .read()
            .await
            .iter()
            .filter(|((idx, _), _)| *idx == layer_idx)
            .map(|(_, shard)| shard.clone())
            .collect()
    }

    /// Total size of cached shards in bytes
    pub async fn cached_size_bytes(&self) -> usize {
        self.shards
            .read()
            .await
            .values()
            .map(|s| s.size_bytes())
            .sum()
    }
}

/// Dequantize a QTensor to f32 vector
/// Note: This is a simplified version - real implementation would preserve quantization
fn dequantize_tensor(qtensor: &QTensor) -> Result<Vec<f32>> {
    // For now, convert to f32 tensor and extract data
    // In production, we'd keep quantized and only dequantize during compute
    let tensor = qtensor.dequantize(&Device::Cpu)?;
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok(data)
}

/// Compress data using gzip (flate2)
fn compress_data(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data).expect("Compression failed");
    encoder.finish().expect("Compression finish failed")
}

/// Decompress data using gzip (flate2)
fn decompress_data(compressed: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(compressed);
    let mut data = Vec::new();
    decoder.read_to_end(&mut data)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_config_mistral() {
        let config = ShardConfig::mistral_7b(4, 2);

        assert_eq!(config.world_size, 4);
        assert_eq!(config.node_rank, 2);
        assert_eq!(config.heads_per_node(), 8); // 32 / 4
        assert_eq!(config.head_range(), (16, 24)); // Rank 2 gets heads 16-23
        assert_eq!(config.ffn_shard_dim(), 3584); // 14336 / 4
    }

    #[test]
    fn test_shard_config_flexible() {
        // Test with 3 nodes (not power of 2)
        let config = ShardConfig::mistral_7b(3, 1);

        assert_eq!(config.heads_per_node(), 10); // 32 / 3 = 10 (floor)
        assert_eq!(config.head_range(), (10, 20));
    }

    #[test]
    fn test_weight_shard_serialization() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shard = WeightShard::new(
            0,
            "attn_q".to_string(),
            data.clone(),
            vec![2, 2],
            1,
            vec![2, 4],
            true,
        );

        let recovered = shard.to_f32_vec().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_temporal_weight_update() {
        let update = TemporalWeightUpdate::new(
            5, // layer_idx
            "attn_q".to_string(),
            vec![0u8; 256], // protected_gradient
            100, // reveal_round
            vec![0u8; 128], // validity_proof
            [42u8; 32], // gradient_checksum
            "node-001".to_string(),
            1.5, // gradient_magnitude
            [1u8; 32], // key_commitment
        );

        assert_eq!(update.layer_idx, 5);
        assert_eq!(update.weight_name, "attn_q");
        assert_eq!(update.reveal_round, 100);
        assert_eq!(update.shares_available, 0);
        assert!(!update.can_apply);
        assert_eq!(update.gradient_magnitude, 1.5);
    }

    #[test]
    fn test_temporal_weight_share_recording() {
        let mut update = TemporalWeightUpdate::new(
            0,
            "ffn_gate".to_string(),
            vec![0u8; 256],
            50,
            vec![0u8; 128],
            [42u8; 32],
            "node-002".to_string(),
            0.5,
            [1u8; 32],
        );

        // Need 3 shares for threshold
        assert!(!update.can_apply);

        update.record_share();
        assert_eq!(update.shares_available, 1);
        assert!(!update.can_apply);

        update.record_share();
        assert_eq!(update.shares_available, 2);
        assert!(!update.can_apply);

        update.record_share();
        assert_eq!(update.shares_available, 3);
        assert!(update.can_apply); // Now we can apply!
    }

    #[test]
    fn test_temporal_weight_magnitude_validation() {
        let update = TemporalWeightUpdate::new(
            0,
            "attn_v".to_string(),
            vec![],
            0,
            vec![],
            [0u8; 32],
            "node".to_string(),
            2.5,
            [0u8; 32],
        );

        // Valid magnitude (within max)
        assert!(update.is_magnitude_valid(10.0));
        assert!(update.is_magnitude_valid(2.5));

        // Invalid (exceeds max)
        assert!(!update.is_magnitude_valid(2.0));

        // Zero magnitude update
        let zero_update = TemporalWeightUpdate::new(
            0,
            "attn_v".to_string(),
            vec![],
            0,
            vec![],
            [0u8; 32],
            "node".to_string(),
            0.0,
            [0u8; 32],
        );
        assert!(!zero_update.is_magnitude_valid(10.0)); // Zero is invalid
    }

    #[test]
    fn test_temporal_weight_update_id() {
        let checksum = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let update = TemporalWeightUpdate::new(
            3,
            "ffn_down".to_string(),
            vec![],
            42,
            vec![],
            checksum,
            "node".to_string(),
            1.0,
            [0u8; 32],
        );

        let id = update.update_id();
        assert!(id.starts_with("3:ffn_down:42:"));
        assert!(id.contains("0102030405060708")); // First 8 bytes of checksum as hex
    }

    #[test]
    fn test_temporal_weight_update_serialization() {
        let update = TemporalWeightUpdate::new(
            7,
            "attn_output".to_string(),
            vec![1u8, 2, 3, 4],
            200,
            vec![5u8, 6, 7, 8],
            [42u8; 32],
            "node-003".to_string(),
            3.14,
            [99u8; 32],
        );

        let bytes = update.to_bytes().unwrap();
        let restored = TemporalWeightUpdate::from_bytes(&bytes).unwrap();

        assert_eq!(update.layer_idx, restored.layer_idx);
        assert_eq!(update.weight_name, restored.weight_name);
        assert_eq!(update.reveal_round, restored.reveal_round);
        assert_eq!(update.gradient_checksum, restored.gradient_checksum);
        assert_eq!(update.source_node, restored.source_node);
        assert_eq!(update.gradient_magnitude, restored.gradient_magnitude);
    }

    #[test]
    fn test_temporal_update_status() {
        assert_eq!(TemporalUpdateStatus::Protected, TemporalUpdateStatus::Protected);
        assert_ne!(TemporalUpdateStatus::Protected, TemporalUpdateStatus::Ready);
        assert_ne!(TemporalUpdateStatus::Applied, TemporalUpdateStatus::Rejected);
    }
}
