# Phase 3: KV-Cache Implementation Roadmap

**Date:** October 28, 2025
**Goal:** 3-5x speedup for autoregressive text generation
**Status:** 🔄 Ready to implement
**Prerequisites:** ✅ Phase 2 complete (full 32-layer generation working)

---

## Overview

KV-cache eliminates redundant computation during autoregressive generation by caching the attention keys and values from previous tokens. This provides 3-5x speedup for multi-token generation.

### Current Performance (Without Cache)
```
Token 1:  45s (compute all 32 layers)
Token 2:  45s (recompute everything!)
Token 3:  45s (recompute everything!)
...
Total for 10 tokens: 450s
```

### Target Performance (With Cache)
```
Token 1:  45s (compute all 32 layers, build cache)
Token 2:  12s (reuse cached K/V, only compute new token)
Token 3:  12s (reuse cached K/V, only compute new token)
...
Total for 10 tokens: 45 + 9*12 = 153s
Speedup: 2.9x (improves with more tokens)
```

---

## Implementation Plan

### Step 1: Add `forward_with_cache()` to MistralAttention

**File:** `crates/q-ai-inference/src/mistral_model.rs`

**Location:** Add after the existing `forward()` method in `impl MistralAttention` (around line 204)

**New Method:**
```rust
/// Forward pass with optional KV-cache for efficient autoregressive generation
pub fn forward_with_cache(
    &self,
    hidden_states: &Tensor,
    attention_mask: Option<&Tensor>,
    position_ids: &Tensor,
    kv_cache: Option<(&Tensor, &Tensor)>, // (cached_k, cached_v)
) -> Result<(Tensor, Tensor, Tensor)> {  // (output, new_k, new_v)
    let (batch_size, seq_len, _) = hidden_states.dims3()?;

    // Project to Q, K, V (only for NEW tokens)
    let flat_hidden = hidden_states.reshape((batch_size * seq_len, 4096))?;

    let q = flat_hidden.matmul(&self.q_proj.t()?)?;
    let q = q.reshape((batch_size, seq_len, 4096))?;

    let k = flat_hidden.matmul(&self.k_proj.t()?)?;
    let k = k.reshape((batch_size, seq_len, 1024))?;

    let v = flat_hidden.matmul(&self.v_proj.t()?)?;
    let v = v.reshape((batch_size, seq_len, 1024))?;

    // Reshape for multi-head attention
    let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?;  // [batch, num_heads, seq_len, head_dim]

    let mut k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
        .transpose(1, 2)?;  // [batch, num_kv_heads, seq_len, head_dim]

    let mut v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
        .transpose(1, 2)?;

    // Apply RoPE to Q and new K
    let (q, k) = self.rope.apply_rotary_emb(&q, &k, position_ids)?;

    // Concatenate with cached K/V if provided
    let (k_full, v_full) = if let Some((cached_k, cached_v)) = kv_cache {
        // Concatenate along sequence dimension (dim=2)
        // cached: [batch, num_kv_heads, cached_seq_len, head_dim]
        // new:    [batch, num_kv_heads, 1, head_dim]
        // result: [batch, num_kv_heads, cached_seq_len+1, head_dim]
        let k_full = Tensor::cat(&[cached_k, &k], 2)?;
        let v_full = Tensor::cat(&[cached_v, &v], 2)?;
        (k_full, v_full)
    } else {
        (k.clone(), v.clone())
    };

    // Store full K/V for next iteration (before GQA expansion)
    let k_cache = k_full.clone();
    let v_cache = v_full.clone();

    // Grouped-Query Attention: repeat K/V heads
    let num_groups = self.num_heads / self.num_kv_heads;
    let k_expanded = k_full.repeat((1, num_groups, 1, 1))?
        .reshape((batch_size, self.num_heads, k_full.dim(2)?, self.head_dim))?;
    let v_expanded = v_full.repeat((1, num_groups, 1, 1))?
        .reshape((batch_size, self.num_heads, v_full.dim(2)?, self.head_dim))?;

    // Scaled dot-product attention
    let scale = 1.0 / (self.head_dim as f64).sqrt();
    let attn_weights = (q.matmul(&k_expanded.t()?)? * scale)?;

    // Apply attention mask if provided (causal mask for autoregressive)
    let attn_weights = if let Some(mask) = attention_mask {
        attn_weights.broadcast_add(mask)?
    } else {
        attn_weights
    };

    // Softmax over last dimension
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Apply attention to values
    let attn_output = attn_weights.matmul(&v_expanded)?;

    // Reshape back to (batch, seq_len, hidden_size)
    let attn_output = attn_output
        .transpose(1, 2)?
        .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

    // Output projection
    let flat_attn = attn_output.reshape((batch_size * seq_len, 4096))?;
    let output = flat_attn.matmul(&self.o_proj.t()?)?;
    let output = output.reshape((batch_size, seq_len, 4096))?;

    Ok((output, k_cache, v_cache))
}
```

**Key Points:**
1. Accepts optional `kv_cache` parameter with previous K/V tensors
2. Computes Q, K, V only for NEW tokens (not full sequence)
3. Concatenates new K/V with cached K/V
4. Returns updated cache for next iteration
5. Maintains compatibility with existing `forward()` method

---

### Step 2: Update MistralLayer to Support Cache

**File:** `crates/q-ai-inference/src/mistral_model.rs`

**Location:** Modify `MistralLayer::forward()` (around line 380)

**Changes Needed:**

1. Add new method `forward_with_cache()`:

```rust
impl MistralLayer {
    /// Forward pass with KV-cache support
    pub fn forward_with_cache(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {  // (output, k_cache, v_cache)
        // Pre-attention normalization
        let normed = self.input_layernorm.forward(hidden_states)?;

        // Self-attention with cache
        let (attn_output, k_cache, v_cache) = self.self_attn.forward_with_cache(
            &normed,
            attention_mask,
            position_ids,
            kv_cache,
        )?;

        // Residual connection
        let hidden_states = (hidden_states + attn_output)?;

        // Pre-FFN normalization
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;

        // Feed-forward
        let ffn_output = self.mlp.forward(&normed)?;

        // Residual connection
        let output = (hidden_states + ffn_output)?;

        Ok((output, k_cache, v_cache))
    }

    // Keep existing forward() for compatibility
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        // Call forward_with_cache with None
        let (output, _, _) = self.forward_with_cache(
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )?;
        Ok(output)
    }
}
```

---

### Step 3: Create Cache Management Structure

**File:** `crates/q-ai-inference/src/kv_cache.rs`

**Add Simple Cache Manager:**

```rust
use std::collections::HashMap;
use candle_core::Tensor;
use anyhow::Result;

/// Simple KV-cache manager for autoregressive generation
pub struct LayerKVCache {
    /// Cached keys per layer: layer_idx -> K tensor
    pub keys: HashMap<usize, Tensor>,
    /// Cached values per layer: layer_idx -> V tensor
    pub values: HashMap<usize, Tensor>,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
        }
    }

    /// Get cached K/V for a layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if let (Some(k), Some(v)) = (self.keys.get(&layer_idx), self.values.get(&layer_idx)) {
            Some((k, v))
        } else {
            None
        }
    }

    /// Update cache for a layer
    pub fn update(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        self.keys.insert(layer_idx, k);
        self.values.insert(layer_idx, v);
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let keys_bytes: usize = self.keys.values()
            .map(|t| t.elem_count() * 4)  // f32 = 4 bytes
            .sum();
        let values_bytes: usize = self.values.values()
            .map(|t| t.elem_count() * 4)
            .sum();
        keys_bytes + values_bytes
    }
}
```

---

### Step 4: Create Multi-Token Generation Example

**File:** `crates/q-ai-inference/examples/test_cached_generation.rs`

**Complete Example:**

```rust
// Cached multi-token generation with KV-cache
//
// This example demonstrates the KV-cache speedup:
// - First token: 45s (full forward pass)
// - Subsequent tokens: 12s each (reuse cache)
// - 3-5x speedup for multi-token generation

use anyhow::Result;
use q_ai_inference::*;
use candle_core::{Device, IndexOp, Tensor};
use std::path::Path;
use std::time::Instant;

const MODEL_PATH: &str = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";
const MAX_NEW_TOKENS: usize = 20;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Q-NarwhalKnight Cached Text Generation");
    println!("==========================================\n");

    if !Path::new(MODEL_PATH).exists() {
        eprintln!("❌ Error: Model not found at: {}", MODEL_PATH);
        return Ok(());
    }

    let total_start = Instant::now();

    // Step 1: Initialize
    println!("📦 Step 1: Loading model...");
    let capability = DeviceCapability::CPU { cores: 8, ram_gb: 16 };
    let model_loader = GGUFModelLoader::new(MODEL_PATH, &capability)?;
    let device = Device::Cpu;
    let tokenizer = GgufTokenizer::from_gguf_file(MODEL_PATH)?;
    let config = MistralConfig::mistral_7b_v0_3();
    println!("   ✅ Model loaded\n");

    // Step 2: Prepare prompt
    println!("✏️  Step 2: Tokenizing prompt...");
    let prompt = "[INST] What is quantum consensus? [/INST]";
    let mut tokens = tokenizer.encode(prompt, false)?;
    println!("   Prompt: {}", prompt);
    println!("   Tokens: {} tokens\n", tokens.len());

    // Step 3: Load model components
    println!("🔧 Step 3: Loading 32 layers...");
    let load_start = Instant::now();
    let special_layers = model_loader.load_special_layers()?;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let layer_weights = model_loader.load_layer(i, &device)?;
        let layer = MistralLayer::from_weights(&layer_weights, &config, &device)?;
        layers.push(layer);
    }
    println!("   ✅ All layers loaded ({:.2}s)\n", load_start.elapsed().as_secs_f64());

    // Step 4: Initialize KV-cache
    println!("💾 Step 4: Initializing KV-cache...");
    let mut kv_cache = LayerKVCache::new();
    println!("   ✅ Cache ready\n");

    // Step 5: Decode embeddings
    let embeddings = special_layers.token_embd.as_ref()
        .ok_or_else(|| anyhow::anyhow!("No embeddings"))?
        .dequantize(&device)?;

    let output_norm_weight = special_layers.output_norm.as_ref()
        .ok_or_else(|| anyhow::anyhow!("No output norm"))?
        .dequantize(&device)?;
    let norm = RMSNorm::new(output_norm_weight, config.rms_norm_eps);

    let output_proj = special_layers.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("No output projection"))?
        .dequantize(&device)?;

    // Step 6: Autoregressive generation
    println!("⚡ Step 6: Generating {} tokens...\n", MAX_NEW_TOKENS);

    let sampling_config = sampling::SamplingConfig::balanced();
    let mut sampler = sampling::Sampler::new(sampling_config);

    let mut generated_tokens = Vec::new();
    let mut token_times = Vec::new();

    for token_idx in 0..MAX_NEW_TOKENS {
        let token_start = Instant::now();

        // Get current sequence length
        let current_seq_len = tokens.len();

        // Create embeddings for NEW token only (last token in sequence)
        let input_ids = Tensor::new(&tokens[tokens.len()-1..], &device)?;
        let mut hidden_states = embeddings.index_select(&input_ids, 0)?;
        hidden_states = hidden_states.reshape((1, 1, 4096))?;  // [batch=1, seq_len=1, hidden=4096]

        // Position IDs for the new token
        let position_ids = Tensor::new(&[current_seq_len as u32 - 1], &device)?
            .reshape((1, 1))?;

        // Forward pass through all 32 layers WITH CACHE
        for (layer_idx, layer) in layers.iter().enumerate() {
            let cache = kv_cache.get(layer_idx);
            let (output, k_cache, v_cache) = layer.forward_with_cache(
                &hidden_states,
                None,  // No attention mask (causal handled in cache concat)
                &position_ids,
                cache,
            )?;

            // Update cache for this layer
            kv_cache.update(layer_idx, k_cache, v_cache);

            hidden_states = output;
        }

        // Final norm + output projection
        hidden_states = norm.forward(&hidden_states)?;
        let last_hidden = hidden_states.i((.., 0, ..))?;  // Get last token
        let logits = last_hidden.matmul(&output_proj.t()?)?;
        let logits_flat = logits.flatten_all()?;

        // Sample next token
        let next_token = sampler.sample(&logits_flat, &tokens)?;

        let token_time = token_start.elapsed();
        token_times.push(token_time.as_secs_f64());

        // Decode and print
        let next_text = tokenizer.decode(&[next_token], true)?;
        println!("   Token {}: \"{}\" ({:.2}s, cache: {:.1}MB)",
            token_idx + 1,
            next_text.trim(),
            token_time.as_secs_f64(),
            kv_cache.memory_bytes() as f64 / 1_000_000.0
        );

        // Add to sequence
        tokens.push(next_token);
        generated_tokens.push(next_token);

        // Check for EOS
        if next_token == tokenizer.eos_token_id() {
            println!("\n   ✅ Reached EOS token");
            break;
        }
    }

    // Step 7: Performance summary
    println!("\n📊 Performance Summary:");
    println!("   - Total time: {:.2}s", total_start.elapsed().as_secs_f64());
    println!("   - Tokens generated: {}", generated_tokens.len());
    println!("   - First token: {:.2}s (with cache build)", token_times[0]);
    if token_times.len() > 1 {
        let avg_cached = token_times[1..].iter().sum::<f64>() / (token_times.len() - 1) as f64;
        println!("   - Avg subsequent: {:.2}s (with cache)", avg_cached);
        println!("   - Speedup: {:.1}x", token_times[0] / avg_cached);
    }
    println!("   - Final cache size: {:.1}MB\n", kv_cache.memory_bytes() as f64 / 1_000_000.0);

    // Step 8: Show generated text
    let generated_text = tokenizer.decode(&generated_tokens, true)?;
    println!("🎉 Generated Text:");
    println!("   Prompt: {}", prompt);
    println!("   Output: {}\n", generated_text);

    println!("✅ CACHED GENERATION SUCCESSFUL!");
    println!("   Expected speedup: 3-5x for longer sequences");

    Ok(())
}
```

---

### Step 5: Add to Cargo.toml

**File:** `crates/q-ai-inference/Cargo.toml`

**Add:**

```toml
[[example]]
name = "test_cached_generation"
path = "examples/test_cached_generation.rs"
```

---

## Expected Results

### Performance Comparison

**Without Cache (Current):**
```
Token 1:  45s
Token 2:  45s
Token 3:  45s
Token 4:  45s
Token 5:  45s
Total:    225s for 5 tokens
```

**With Cache (Target):**
```
Token 1:  45s (build cache)
Token 2:  12s (use cache)
Token 3:  12s (use cache)
Token 4:  12s (use cache)
Token 5:  12s (use cache)
Total:    93s for 5 tokens
Speedup:  2.4x
```

**With Cache (10 tokens):**
```
Token 1:   45s
Tokens 2-10: 9 × 12s = 108s
Total:     153s
vs without: 450s
Speedup:   2.9x
```

**With Cache (50 tokens):**
```
Token 1:    45s
Tokens 2-50: 49 × 12s = 588s
Total:      633s
vs without: 2250s
Speedup:    3.6x
```

### Memory Usage

| Tokens | Cache Size | Total RAM |
|--------|------------|-----------|
| 1      | 0 MB       | 5 GB      |
| 10     | ~200 MB    | 5.2 GB    |
| 50     | ~1 GB      | 6 GB      |
| 100    | ~2 GB      | 7 GB      |

---

## Testing Plan

### Test 1: Single Token with Cache Build
```bash
# Verify cache creation works
cargo run --package q-ai-inference --example test_cached_generation --release
```

**Expected Output:**
```
Token 1: "Quantum" (45.2s, cache: 0.0MB)
Token 2: "consensus" (12.1s, cache: 42.3MB)
Token 3: "is" (11.9s, cache: 84.6MB)
...
```

### Test 2: Performance Validation
```bash
# Compare with non-cached version
# Should see 3-5x speedup after first token
```

### Test 3: Memory Validation
```bash
# Monitor memory usage
# Cache should be ~2GB per 1000 tokens
```

---

## Implementation Checklist

- [ ] Add `forward_with_cache()` to `MistralAttention`
- [ ] Update `MistralLayer::forward()` to support cache
- [ ] Create `LayerKVCache` manager
- [ ] Create `test_cached_generation.rs` example
- [ ] Add example to Cargo.toml
- [ ] Test cache build (first token)
- [ ] Test cache reuse (subsequent tokens)
- [ ] Verify 3-5x speedup
- [ ] Validate memory usage
- [ ] Document performance improvements

---

## Common Issues & Solutions

### Issue 1: Tensor Shape Mismatch
**Problem:** Cached K/V shape doesn't match new K/V
**Solution:** Ensure concatenation happens on correct dimension (dim=2 for sequence length)

### Issue 2: Position IDs Out of Range
**Problem:** Position IDs don't account for cached sequence length
**Solution:** Position IDs should be `[cached_len + new_token_idx]`

### Issue 3: Memory Growth
**Problem:** Cache grows unbounded
**Solution:** Implement cache eviction or sliding window

### Issue 4: Slower Than Expected
**Problem:** Not seeing 3-5x speedup
**Solution:** Ensure only computing Q/K/V for NEW tokens, not full sequence

---

## Next Steps After KV-Cache

Once KV-cache is working:

1. **Autoregressive Loop** - Generate complete responses
2. **Stop Conditions** - Detect EOS token, max length
3. **Streaming Output** - Display tokens as generated
4. **Distributed Inference** - Split layers across libp2p
5. **Privacy Layer** - Add AEGIS-QL + ZK-STARK

---

**This roadmap provides complete implementation details for Phase 3!**
When ready to implement, follow each step sequentially.

Expected time to complete: 1-2 days
Expected speedup: 3-5x for multi-token generation
Expected complexity: Medium (straightforward tensor concatenation)
