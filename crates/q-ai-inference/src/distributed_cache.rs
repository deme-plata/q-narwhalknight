// Distributed inference with KV-cache integration
// Enables multi-token generation across P2P network with per-layer caching

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{
    simple_kv_cache::LayerKVCache,
    mistral_model::{MistralLayer, MistralConfig, RMSNorm},
    gguf_loader::GGUFModelLoader,
    tokenizer::GgufTokenizer,
    sampling::{Sampler, SamplingConfig},
};
use candle_core::quantized::QTensor;

/// Distributed inference engine with KV-cache support
/// Designed for P2P layer distribution across multiple nodes
pub struct DistributedInferenceWithCache {
    /// All transformer layers (may be distributed across nodes in future)
    layers: Vec<MistralLayer>,

    /// KV-cache per layer (one cache per transformer layer)
    caches: Vec<LayerKVCache>,

    /// Special layers (embeddings, output projection, norms)
    special_layers: Arc<SpecialLayers>,

    /// Tokenizer for encoding/decoding
    tokenizer: GgufTokenizer,

    /// Model configuration
    config: MistralConfig,

    /// Compute device
    device: Device,

    /// Sampler for next-token prediction
    sampler: Sampler,

    /// Statistics tracking
    stats: Arc<RwLock<InferenceStats>>,
}

/// Special layers that aren't part of transformers
pub struct SpecialLayers {
    pub token_embd: Option<QTensor>,
    pub output: Option<QTensor>,
    pub output_norm: Option<QTensor>,
}

/// Statistics for inference performance monitoring
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    pub total_tokens_generated: usize,
    pub total_generation_time_ms: f32,
    pub average_time_per_token_ms: f32,
    pub cache_hit_count: usize,
    pub cache_miss_count: usize,
    pub speedup_factor: f32,
}

impl DistributedInferenceWithCache {
    /// Create a new distributed inference engine with KV-cache
    pub async fn new(model_path: &str, config: MistralConfig, device: Device) -> Result<Self> {
        let tokenizer = GgufTokenizer::from_gguf_file(model_path)?;

        let capability = crate::DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };

        let model_loader = GGUFModelLoader::new(model_path, &capability)?;

        // Load special layers (load once, QTensor doesn't support clone)
        let loaded_special = model_loader.load_special_layers()?;
        let special_layers = Arc::new(SpecialLayers {
            token_embd: loaded_special.token_embd,
            output: loaded_special.output,
            output_norm: loaded_special.output_norm,
        });

        // Load all transformer layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer_weights = model_loader.load_layer(i, &device)?;
            let layer = MistralLayer::from_weights(&layer_weights, &config, &device)?;
            layers.push(layer);
        }

        // Initialize KV-cache for each layer
        let caches = (0..config.num_hidden_layers)
            .map(|_| LayerKVCache::new())
            .collect();

        let sampling_config = SamplingConfig::balanced();
        let sampler = Sampler::new(sampling_config);

        Ok(Self {
            layers,
            caches,
            special_layers,
            tokenizer,
            config,
            device,
            sampler,
            stats: Arc::new(RwLock::new(InferenceStats::default())),
        })
    }

    /// Generate text with KV-cache acceleration
    ///
    /// # Arguments
    /// * `prompt` - Input text to continue from
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Generated text continuation
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Reset cache for new generation
        self.reset_cache();

        // Encode prompt
        let token_ids = self.tokenizer.encode(prompt, false)?;
        let mut current_tokens = token_ids.clone();
        let mut generated_tokens = Vec::new();

        let generation_start = std::time::Instant::now();
        let mut step_timings = Vec::new();

        // Generate tokens one by one
        for step in 0..max_tokens {
            let step_start = std::time::Instant::now();

            // For first token: use all input tokens
            // For subsequent tokens: use only the new token (cache provides context)
            let input_for_step = if step == 0 {
                current_tokens.clone()
            } else {
                vec![*current_tokens.last().unwrap()]
            };

            // Forward pass with cache
            let next_token = self.forward_with_cache(&input_for_step, &current_tokens)?;

            let step_time = step_start.elapsed().as_secs_f32() * 1000.0; // ms
            step_timings.push(step_time);

            // Append generated token
            current_tokens.push(next_token);
            generated_tokens.push(next_token);

            // Check for EOS token (optional - model dependent)
            // if next_token == EOS_TOKEN_ID { break; }
        }

        let total_time = generation_start.elapsed().as_secs_f32() * 1000.0; // ms

        // Update statistics
        self.update_stats(generated_tokens.len(), total_time, &step_timings).await;

        // Decode generated tokens
        let full_text = self.tokenizer.decode(&current_tokens, false)?;
        Ok(full_text)
    }

    /// Generate text with KV-cache, yielding tokens one-by-one for streaming
    ///
    /// This is designed for Server-Sent Events (SSE) streaming where each token
    /// is sent to the client immediately as it's generated.
    ///
    /// # Arguments
    /// * `prompt` - Input text to continue from
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `callback` - Async closure called for each generated token
    ///
    /// # Returns
    /// Final generated text
    ///
    /// # Example
    /// ```rust
    /// engine.generate_stream(
    ///     "[INST] Hello [/INST]",
    ///     50,
    ///     |token_id, token_text, position| async move {
    ///         println!("Token {}: {} (ID: {})", position, token_text, token_id);
    ///         Ok(())
    ///     }
    /// ).await?;
    /// ```
    pub async fn generate_stream<F, Fut>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(u32, String, usize) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        // Reset cache for new generation
        self.reset_cache();

        // Encode prompt
        let token_ids = self.tokenizer.encode(prompt, false)?;
        let mut current_tokens = token_ids.clone();
        let mut generated_tokens = Vec::new();

        let generation_start = std::time::Instant::now();
        let mut step_timings = Vec::new();

        // Generate tokens one by one, calling callback for each
        for step in 0..max_tokens {
            let step_start = std::time::Instant::now();

            // For first token: use all input tokens
            // For subsequent tokens: use only the new token (cache provides context)
            let input_for_step = if step == 0 {
                current_tokens.clone()
            } else {
                vec![*current_tokens.last().unwrap()]
            };

            // Forward pass with cache
            let next_token = self.forward_with_cache(&input_for_step, &current_tokens)?;

            let step_time = step_start.elapsed().as_secs_f32() * 1000.0; // ms
            step_timings.push(step_time);

            // Decode this single token to text
            let token_text = self.tokenizer.decode(&[next_token], false)?;

            // Append generated token
            current_tokens.push(next_token);
            generated_tokens.push(next_token);

            // Call callback with this token (for SSE streaming)
            callback(next_token, token_text, step).await?;

            // Check for EOS token (Mistral uses token_id 2)
            if next_token == 2 {
                break;
            }
        }

        let total_time = generation_start.elapsed().as_secs_f32() * 1000.0; // ms

        // Update statistics
        self.update_stats(generated_tokens.len(), total_time, &step_timings).await;

        // Decode all generated tokens
        let full_text = self.tokenizer.decode(&current_tokens, false)?;
        Ok(full_text)
    }

    /// Forward pass through all layers with KV-cache
    fn forward_with_cache(
        &mut self,
        input_tokens: &[u32],
        all_tokens: &[u32],
    ) -> Result<u32> {
        // Create embeddings
        let input_tensor = Tensor::new(input_tokens, &self.device)?;
        let embeddings = self.special_layers.token_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No embeddings"))?
            .dequantize(&self.device)?;

        let mut hidden_states = embeddings.index_select(&input_tensor, 0)?;
        let batch_size = 1;
        let seq_len = input_tokens.len();
        hidden_states = hidden_states.reshape((batch_size, seq_len, self.config.hidden_size))?;

        // Create position IDs (absolute positions!)
        let start_pos = all_tokens.len() - input_tokens.len();
        let position_ids: Vec<u32> = (start_pos..(start_pos + seq_len))
            .map(|p| p as u32)
            .collect();
        let position_ids_tensor = Tensor::new(&position_ids[..], &self.device)?;

        // Forward through all layers WITH CACHE
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_cache(
                &hidden_states,
                None, // attention mask (optional)
                &position_ids_tensor,
                Some(&mut self.caches[layer_idx]), // KV-cache for this layer
            )?;
        }

        // Apply final norm
        let norm_weight = self.special_layers.output_norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No output_norm"))?
            .dequantize(&self.device)?;
        let norm = RMSNorm::new(norm_weight, self.config.rms_norm_eps);
        hidden_states = norm.forward(&hidden_states)?;

        // Extract last token's hidden state
        let last_hidden = hidden_states.i((.., seq_len - 1, ..))?;

        // Output projection
        let output = self.special_layers.output.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No output layer"))?
            .dequantize(&self.device)?;
        let last_logits = last_hidden.matmul(&output.t()?)?;

        // Sample next token
        let last_logits_flat = last_logits.squeeze(0)?;
        let next_token = self.sampler.sample(&last_logits_flat, all_tokens)?;

        Ok(next_token)
    }

    /// Reset KV-cache for all layers
    pub fn reset_cache(&mut self) {
        for cache in &mut self.caches {
            *cache = LayerKVCache::new();
        }
    }

    /// Get current cache size (tokens cached)
    pub fn cache_size(&self) -> usize {
        self.caches.first().map(|c| c.cache_size()).unwrap_or(0)
    }

    /// Update statistics
    async fn update_stats(&self, tokens_generated: usize, total_time_ms: f32, step_timings: &[f32]) {
        let mut stats = self.stats.write().await;

        stats.total_tokens_generated += tokens_generated;
        stats.total_generation_time_ms += total_time_ms;
        stats.average_time_per_token_ms = total_time_ms / tokens_generated as f32;

        // Calculate speedup factor (baseline vs cached)
        if step_timings.len() > 1 {
            let baseline = step_timings[0];
            let avg_cached = step_timings[1..].iter().sum::<f32>() / (step_timings.len() - 1) as f32;
            stats.speedup_factor = baseline / avg_cached;

            stats.cache_hit_count += step_timings.len() - 1;
            stats.cache_miss_count += 1;
        }
    }

    /// Get inference statistics
    pub async fn get_stats(&self) -> InferenceStats {
        self.stats.read().await.clone()
    }

    /// Get model configuration
    pub fn config(&self) -> &MistralConfig {
        &self.config
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Future: Distributed layer execution across P2P network
///
/// This will enable:
/// - Splitting layers across multiple nodes (e.g., 4 nodes × 8 layers each)
/// - Passing hidden states between nodes via libp2p
/// - Maintaining per-layer caches on each node
/// - Load balancing based on node capabilities
#[allow(dead_code)]
pub struct DistributedLayerExecutor {
    /// Node assignment: which layers this node is responsible for
    local_layer_indices: Vec<usize>,

    /// Local layers and caches
    local_layers: Vec<MistralLayer>,
    local_caches: Vec<LayerKVCache>,

    // P2P network interface (future integration)
    // network: P2PNetwork,

    // Next node in the layer chain (to send hidden states to)
    // next_node: Option<NodeId>,
}

impl DistributedLayerExecutor {
    /// Execute assigned layers and pass hidden states to next node
    #[allow(dead_code)]
    pub async fn execute_layers(
        &mut self,
        hidden_states: Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let mut current_hidden = hidden_states;

        // Execute local layers with cache
        for (i, layer) in self.local_layers.iter().enumerate() {
            current_hidden = layer.forward_with_cache(
                &current_hidden,
                None,
                position_ids,
                Some(&mut self.local_caches[i]),
            )?;
        }

        // Future: Send hidden states to next node via libp2p
        // if let Some(next) = &self.next_node {
        //     self.network.send_hidden_states(next, &current_hidden).await?;
        // }

        Ok(current_hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_inference_creation() {
        // Test that we can create the inference engine
        // (requires model file to exist)
        let model_path = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

        if std::path::Path::new(model_path).exists() {
            let config = MistralConfig::mistral_7b_v0_3();
            let device = Device::Cpu;

            let result = DistributedInferenceWithCache::new(model_path, config, device).await;
            assert!(result.is_ok(), "Should create inference engine");

            let engine = result.unwrap();
            assert_eq!(engine.num_layers(), 32, "Should have 32 layers");
            assert_eq!(engine.cache_size(), 0, "Cache should be empty initially");
        }
    }
}
