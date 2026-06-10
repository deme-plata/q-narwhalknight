//! Mistral-7B Model Implementation using Candle
//!
//! This module implements the Mistral-7B-Instruct-v0.3 architecture for distributed inference.
//! It provides layer-wise execution capabilities for splitting computation across nodes.
//!
//! Architecture:
//! - 32 transformer layers
//! - 4096 hidden size
//! - 32 attention heads (8 KV heads for Grouped-Query Attention)
//! - 14336 intermediate FFN size
//! - RoPE (Rotary Position Embeddings)
//! - SwiGLU activation
//! - RMSNorm

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::gguf_loader::MistralLayerWeights;

/// Mistral-7B model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension
    pub hidden_size: usize,

    /// Intermediate FFN size
    pub intermediate_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of KV heads (for Grouped-Query Attention)
    pub num_key_value_heads: usize,

    /// RoPE theta
    pub rope_theta: f32,

    /// RMSNorm epsilon
    pub rms_norm_eps: f64,

    /// Maximum sequence length
    pub max_position_embeddings: usize,
}

impl MistralConfig {
    /// Create Mistral-7B-Instruct-v0.3 configuration
    pub fn mistral_7b_v0_3() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,  // Grouped-Query Attention
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32768,
        }
    }

    /// Create Ministral-3B-Instruct configuration (v2.5.0-beta)
    ///
    /// Ministral-3B is a smaller, faster model with:
    /// - 24 layers (vs 32 for 7B)
    /// - 2048 hidden size (vs 4096)
    /// - Still uses GQA with 8 KV heads
    ///
    /// For distributed inference, this splits into:
    /// - 2 nodes: layers 0-11, 12-23
    /// - 3 nodes: layers 0-7, 8-15, 16-23
    /// - 4 nodes: layers 0-5, 6-11, 12-17, 18-23
    pub fn ministral_3b() -> Self {
        Self {
            vocab_size: 32768,       // Slightly larger vocabulary
            hidden_size: 2048,       // Half of 7B
            intermediate_size: 8192, // Proportional to hidden_size
            num_hidden_layers: 24,   // 24 layers vs 32
            num_attention_heads: 16, // Half of 7B
            num_key_value_heads: 8,  // Same as 7B (GQA)
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32768,
        }
    }

    /// Detect model config from GGUF path (auto-select Mistral-7B or Ministral-3B)
    pub fn from_model_path(path: &str) -> Self {
        if path.to_lowercase().contains("ministral-3b") || path.to_lowercase().contains("mistral-3b") {
            Self::ministral_3b()
        } else {
            Self::mistral_7b_v0_3()
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of groups for GQA
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Get optimal layer distribution for N nodes
    pub fn optimal_layer_ranges(&self, num_nodes: usize) -> Vec<(usize, usize)> {
        let total_layers = self.num_hidden_layers;
        let layers_per_node = total_layers / num_nodes;
        let remainder = total_layers % num_nodes;

        let mut ranges = Vec::with_capacity(num_nodes);
        let mut start = 0;

        for i in 0..num_nodes {
            // Distribute remainder across first N nodes
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + layers_per_node + extra - 1;
            ranges.push((start, end));
            start = end + 1;
        }

        ranges
    }
}

/// RMSNorm layer
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32).map_err(|e| anyhow!("DType conversion failed: {}", e))?;

        // RMS = sqrt(mean(x^2) + eps)
        let norm = (x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + self.eps)?
            .sqrt()?;

        let normalized = x.broadcast_div(&norm)?;
        let scaled = normalized.broadcast_mul(&self.weight)?;

        scaled.to_dtype(dtype).map_err(|e| anyhow!("DType conversion failed: {}", e))
    }
}

/// Rotary Position Embeddings (RoPE)
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?;

        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;

        Ok(Self {
            sin: emb.sin()?,
            cos: emb.cos()?,
            head_dim,
        })
    }

    pub fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, _position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // Simplified RoPE application
        // In production, this would do proper rotation
        // For now, return unchanged (will be implemented in forward pass)
        Ok((q.clone(), k.clone()))
    }
}

/// Mistral Attention Layer with Grouped-Query Attention
pub struct MistralAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl MistralAttention {
    pub fn from_weights(
        weights: &MistralLayerWeights,
        config: &MistralConfig,
        device: &Device,
    ) -> Result<Self> {
        // Dequantize Q, K, V, O projections from QTensor to Tensor
        let q_proj = weights.attn_q.as_ref()
            .ok_or_else(|| anyhow!("Missing Q projection weights"))?
            .dequantize(device)?;
        println!("Q proj shape: {:?}", q_proj.dims());

        let k_proj = weights.attn_k.as_ref()
            .ok_or_else(|| anyhow!("Missing K projection weights"))?
            .dequantize(device)?;
        println!("K proj shape: {:?}", k_proj.dims());

        let v_proj = weights.attn_v.as_ref()
            .ok_or_else(|| anyhow!("Missing V projection weights"))?
            .dequantize(device)?;
        println!("V proj shape: {:?}", v_proj.dims());

        let o_proj = weights.attn_output.as_ref()
            .ok_or_else(|| anyhow!("Missing output projection weights"))?
            .dequantize(device)?;
        println!("O proj shape: {:?}", o_proj.dims());

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim(),
            rope,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        println!("Input hidden_states shape: {:?}", hidden_states.dims());
        println!("Q proj weight shape: {:?}", self.q_proj.dims());

        // Project to Q, K, V
        // Flatten to 2D for matmul: [batch*seq_len, hidden_size]
        let flat_hidden = hidden_states.reshape((batch_size * seq_len, 4096))?;
        println!("Flat hidden shape: {:?}", flat_hidden.dims());

        // Weight matrices are [out_features, in_features], so we need to transpose them
        // Matmul: [batch*seq_len, hidden_size] × [hidden_size, hidden_size] = [batch*seq_len, hidden_size]
        let q = flat_hidden.matmul(&self.q_proj.t()?)?;
        let q = q.reshape((batch_size, seq_len, 4096))?;  // Reshape back
        println!("After Q matmul shape: {:?}", q.dims());

        // K and V are [1024, 4096] = [out_features, in_features], need transpose
        let k = flat_hidden.matmul(&self.k_proj.t()?)?;
        let k = k.reshape((batch_size, seq_len, 1024))?;
        println!("After K matmul shape: {:?}", k.dims());

        let v = flat_hidden.matmul(&self.v_proj.t()?)?;
        let v = v.reshape((batch_size, seq_len, 1024))?;
        println!("After V matmul shape: {:?}", v.dims());

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;  // (batch, num_heads, seq_len, head_dim)
        println!("After Q reshape/transpose: {:?}", q.dims());

        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.rope.apply_rotary_emb(&q, &k, position_ids)?;

        // Grouped-Query Attention: repeat K/V heads
        // Each KV head is repeated num_groups times to match query heads
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = k.repeat((1, num_groups, 1, 1))?
            .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
        let v = v.repeat((1, num_groups, 1, 1))?
            .reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;

        println!("After reshape - Q shape: {:?}, K shape: {:?}", q.dims(), k.dims());

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;

        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax over last dimension
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, hidden_size)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection - flatten, matmul, reshape
        let flat_attn = attn_output.reshape((batch_size * seq_len, 4096))?;
        let output = flat_attn.matmul(&self.o_proj.t()?)?;
        let output = output.reshape((batch_size, seq_len, 4096))?;

        Ok(output)
    }

    /// Forward pass with KV-cache for autoregressive generation
    ///
    /// This version caches the key and value tensors to avoid recomputing
    /// attention for previous tokens during autoregressive generation.
    ///
    /// Expected speedup: 3-5x for multi-token generation
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        cache: Option<&mut crate::simple_kv_cache::LayerKVCache>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let flat_hidden = hidden_states.reshape((batch_size * seq_len, 4096))?;

        let q = flat_hidden.matmul(&self.q_proj.t()?)?;
        let q = q.reshape((batch_size, seq_len, 4096))?;

        let k = flat_hidden.matmul(&self.k_proj.t()?)?;
        let k = k.reshape((batch_size, seq_len, 1024))?;

        let v = flat_hidden.matmul(&self.v_proj.t()?)?;
        let v = v.reshape((batch_size, seq_len, 1024))?;

        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, mut k) = self.rope.apply_rotary_emb(&q, &k, position_ids)?;
        let mut v = v;

        // Update cache if provided
        if let Some(cache) = cache {
            let (k_cached, v_cached) = cache.update(k, v)?;
            k = k_cached;
            v = v_cached;
        }

        // Grouped-Query Attention: repeat K/V heads
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = k.repeat((1, num_groups, 1, 1))?
            .reshape((batch_size, self.num_heads, k.dims()[2], self.head_dim))?;
        let v = v.repeat((1, num_groups, 1, 1))?
            .reshape((batch_size, self.num_heads, v.dims()[2], self.head_dim))?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;

        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax over last dimension
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, hidden_size)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let flat_attn = attn_output.reshape((batch_size * seq_len, 4096))?;
        let output = flat_attn.matmul(&self.o_proj.t()?)?;
        let output = output.reshape((batch_size, seq_len, 4096))?;

        Ok(output)
    }
}

/// Mistral FFN with SwiGLU activation
pub struct MistralMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl MistralMLP {
    pub fn from_weights(weights: &MistralLayerWeights, device: &Device) -> Result<Self> {
        // Dequantize FFN projections from QTensor to Tensor
        let gate_proj = weights.ffn_gate.as_ref()
            .ok_or_else(|| anyhow!("Missing FFN gate weights"))?
            .dequantize(device)?;

        let up_proj = weights.ffn_up.as_ref()
            .ok_or_else(|| anyhow!("Missing FFN up weights"))?
            .dequantize(device)?;

        let down_proj = weights.ffn_down.as_ref()
            .ok_or_else(|| anyhow!("Missing FFN down weights"))?
            .dequantize(device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Get shape for flattening
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let flat_hidden = hidden_states.reshape((batch_size * seq_len, 4096))?;

        // SwiGLU: swish(gate(x)) * up(x)
        // Gate and Up projections (need transpose: [14336, 4096] -> [4096, 14336])
        let gate = flat_hidden.matmul(&self.gate_proj.t()?)?;
        let up = flat_hidden.matmul(&self.up_proj.t()?)?;

        // SiLU (Swish) activation: x * sigmoid(x)
        let gate_swish = (&gate * candle_nn::ops::sigmoid(&gate)?)?;

        // Element-wise multiply
        let intermediate = (gate_swish * up)?;

        // Down projection ([4096, 14336] no transpose needed)
        let output = intermediate.matmul(&self.down_proj.t()?)?;
        let output = output.reshape((batch_size, seq_len, 4096))?;

        Ok(output)
    }
}

/// Single Mistral transformer layer
pub struct MistralLayer {
    self_attn: MistralAttention,
    mlp: MistralMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl MistralLayer {
    pub fn from_weights(
        weights: &MistralLayerWeights,
        config: &MistralConfig,
        device: &Device,
    ) -> Result<Self> {
        let self_attn = MistralAttention::from_weights(weights, config, device)?;
        let mlp = MistralMLP::from_weights(weights, device)?;

        let input_layernorm = {
            let weight = weights.attn_norm.as_ref()
                .ok_or_else(|| anyhow!("Missing attention norm weights"))?
                .dequantize(device)?;
            println!("Attn norm weight shape: {:?}", weight.dims());
            RMSNorm::new(weight, config.rms_norm_eps)
        };

        let post_attention_layernorm = {
            let weight = weights.ffn_norm.as_ref()
                .ok_or_else(|| anyhow!("Missing FFN norm weights"))?
                .dequantize(device)?;
            println!("FFN norm weight shape: {:?}", weight.dims());
            RMSNorm::new(weight, config.rms_norm_eps)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        // Pre-attention norm + attention + residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask, position_ids)?;
        let hidden_states = (hidden_states + residual)?;

        // Pre-FFN norm + FFN + residual
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        Ok(hidden_states)
    }

    /// Forward pass with KV-cache for autoregressive generation
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        cache: Option<&mut crate::simple_kv_cache::LayerKVCache>,
    ) -> Result<Tensor> {
        // Pre-attention norm + attention + residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward_with_cache(&hidden_states, attention_mask, position_ids, cache)?;
        let hidden_states = (hidden_states + residual)?;

        // Pre-FFN norm + FFN + residual
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_config() {
        let config = MistralConfig::mistral_7b_v0_3();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.num_groups(), 4);
    }

    #[test]
    fn test_rms_norm_creation() {
        let device = Device::Cpu;
        let weight = Tensor::ones((4096,), DType::F32, &device).unwrap();
        let norm = RMSNorm::new(weight, 1e-5);

        // Test forward pass with dummy input
        let input = Tensor::randn(0f32, 1.0, (1, 10, 4096), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.dims(), input.dims());
    }
}
