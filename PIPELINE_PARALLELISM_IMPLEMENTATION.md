# Pipeline Parallelism Implementation - TRUE Distributed Inference

## 🎯 **Goal: 4 Nodes Process 1 Request 4× Faster**

**Date**: November 6, 2025
**Version**: v0.9.27-beta
**Approach**: Custom layer-by-layer execution using existing Mistral model

---

## 🏗️ **Architecture Overview**

### **Pipeline Parallelism (Layer Splitting)**:
```
         Prompt "Hello"
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│         Node 1: Input Embedding + Layers 0-7              │
│  [batch=1, seq_len=5, hidden=4096] tensor                │
└─────────────────────┬────────────────────────────────────┘
                      │ Hidden states over network
                      ▼
┌──────────────────────────────────────────────────────────┐
│         Node 2: Layers 8-15                               │
│  [batch=1, seq_len=5, hidden=4096] tensor                │
└─────────────────────┬────────────────────────────────────┘
                      │ Hidden states over network
                      ▼
┌──────────────────────────────────────────────────────────┐
│         Node 3: Layers 16-23                              │
│  [batch=1, seq_len=5, hidden=4096] tensor                │
└─────────────────────┬────────────────────────────────────┘
                      │ Hidden states over network
                      ▼
┌──────────────────────────────────────────────────────────┐
│         Node 4: Layers 24-31 + LM Head → Logits          │
│  [batch=1, vocab=32000] → Sample → "world"               │
└──────────────────────────────────────────────────────────┘
```

### **Performance Benefits**:
- **Lower Latency per Token**: 4 nodes process layers in parallel pipeline
- **Memory Efficiency**: Each node loads only 8/32 layers (25% of model)
- **Scalability**: Can split Mistral-Small-3.2-24B across more nodes

---

## 📦 **What We Already Have**

### ✅ **Existing Infrastructure** (Ready to Use):

1. **`crates/q-ai-inference/src/mistral_model.rs`**:
   - ✅ `MistralLayer::forward()` - Single layer execution
   - ✅ `MistralLayer::forward_with_cache()` - With KV-cache
   - ✅ `RMSNorm`, `RotaryEmbedding`, `MistralAttention`, `MistralMLP`
   - ✅ Full Mistral-7B architecture

2. **`crates/q-ai-inference/src/gguf_loader.rs`**:
   - ✅ GGUF file parsing
   - ✅ Load weights for specific layers
   - ✅ `GGUFModelLoader::load_layers(start, end)`

3. **`crates/q-network/src/layer_forwarding.rs`**:
   - ✅ `TensorData` - Serializable tensor format
   - ✅ `LayerOutputManager` - Manages tensor forwarding

4. **`crates/q-network/src/distributed_ai_worker.rs`**:
   - ✅ Worker structure
   - ✅ Gossipsub message handling
   - ⚠️  Placeholder layer execution (needs real implementation)

---

## 🔧 **Implementation Plan**

### **Step 1: Complete `DistributedMistralEngine`** ⭐ (Critical)

**File**: `crates/q-ai-inference/src/distributed_engine.rs`

**Current Status**: Stub with placeholder methods

**What to Implement**:

```rust
pub struct DistributedMistralEngine {
    /// Loaded model layers (only assigned range)
    layers: Vec<MistralLayer>,

    /// Model configuration
    config: MistralConfig,

    /// Device (CPU or GPU)
    device: Device,

    /// Assigned layer range
    layer_range: (usize, usize),

    /// Tokenizer
    tokenizer: Arc<Tokenizer>,

    /// Input embedding layer (if first node)
    input_embedding: Option<Tensor>,  // [vocab_size, hidden_size]

    /// Output LM head (if last node)
    lm_head: Option<Tensor>,  // [hidden_size, vocab_size]
}

impl DistributedMistralEngine {
    /// Load model shard for assigned layers
    pub async fn load_from_gguf(
        model_path: &str,
        tokenizer_path: &str,
        layer_range: (usize, usize),
    ) -> Result<Self> {
        let config = MistralConfig::mistral_7b_v0_3();
        let device = Device::Cpu; // TODO: GPU support

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // Load GGUF and extract weights for assigned layers
        let gguf_loader = GGUFModelLoader::new(model_path)?;
        let layer_weights = gguf_loader.load_layers(layer_range.0, layer_range.1)?;

        // Build MistralLayer instances
        let mut layers = Vec::new();
        for weights in layer_weights {
            let layer = MistralLayer::from_weights(&weights, &config, &device)?;
            layers.push(layer);
        }

        // Load embeddings if first node (layer_range.0 == 0)
        let input_embedding = if layer_range.0 == 0 {
            Some(gguf_loader.load_embedding()?)
        } else {
            None
        };

        // Load LM head if last node (layer_range.1 == 31)
        let lm_head = if layer_range.1 == 31 {
            Some(gguf_loader.load_lm_head()?)
        } else {
            None
        };

        Ok(Self {
            layers,
            config,
            device,
            layer_range,
            tokenizer: Arc::new(tokenizer),
            input_embedding,
            lm_head,
        })
    }

    /// Execute assigned layers
    pub async fn execute_layers(
        &self,
        input_hidden: Vec<f32>,
        input_shape: Vec<usize>,
        position_ids: Vec<u32>,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        // Convert to Tensor
        let mut hidden_states = Tensor::from_vec(
            input_hidden,
            input_shape.as_slice(),
            &self.device,
        )?;

        // Create position_ids tensor
        let pos_ids = Tensor::new(position_ids.as_slice(), &self.device)?;

        // Execute each layer sequentially
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_idx = self.layer_range.0 + i;
            debug!("⚙️  Executing layer {}", layer_idx);

            hidden_states = layer.forward(
                &hidden_states,
                None, // attention_mask
                &pos_ids,
            )?;
        }

        // If last node, apply LM head and return logits
        if let Some(ref lm_head) = self.lm_head {
            hidden_states = hidden_states.matmul(&lm_head.t()?)?;
        }

        // Convert back to Vec<f32>
        let output_shape = hidden_states.dims().to_vec();
        let output_data = hidden_states.to_vec1::<f32>()?;

        Ok((output_data, output_shape))
    }

    /// Get embeddings from tokens (first node only)
    pub async fn get_embeddings(&self, prompt: &str) -> Result<(Vec<f32>, Vec<usize>, Vec<u32>)> {
        if self.input_embedding.is_none() {
            return Err(anyhow!("Not first node - no embeddings available"));
        }

        // Tokenize
        let encoding = self.tokenizer.encode(prompt, false)?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Get embeddings: embedding[input_ids]
        let input_ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)?;
        let embeddings = self.input_embedding.as_ref().unwrap()
            .index_select(&input_ids_tensor, 0)?;

        let shape = embeddings.dims().to_vec();
        let data = embeddings.to_vec1::<f32>()?;

        Ok((data, shape, input_ids))
    }
}
```

---

### **Step 2: Update Worker to Use Real Engine** ⭐

**File**: `crates/q-network/src/distributed_ai_worker.rs`

**Changes Needed**:

```rust
pub struct DistributedAIWorker {
    coordinator: Arc<DistributedAICoordinator>,
    active_requests: Arc<RwLock<HashMap<String, ActiveInferenceRequest>>>,

    // NEW: Add DistributedMistralEngine
    engine: Option<Arc<DistributedMistralEngine>>,
    assigned_layers: Option<(usize, usize)>,
}

impl DistributedAIWorker {
    /// Initialize engine with assigned layer range
    pub async fn initialize_engine(
        &mut self,
        model_path: &str,
        tokenizer_path: &str,
        layer_range: (usize, usize),
    ) -> Result<()> {
        info!("🔧 Loading model layers {}-{}", layer_range.0, layer_range.1);

        let engine = DistributedMistralEngine::load_from_gguf(
            model_path,
            tokenizer_path,
            layer_range,
        ).await?;

        self.engine = Some(Arc::new(engine));
        self.assigned_layers = Some(layer_range);

        info!("✅ Engine initialized with {} layers", layer_range.1 - layer_range.0 + 1);
        Ok(())
    }

    /// Execute inference on assigned layers (REAL IMPLEMENTATION)
    async fn run_model_layers(
        &self,
        input_tensor: TensorData,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<TensorData> {
        let engine = self.engine.as_ref()
            .ok_or_else(|| anyhow!("Engine not initialized"))?;

        // Extract position IDs (assume sequential for now)
        let seq_len = input_tensor.shape[1];
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();

        // Execute layers
        let (output_data, output_shape) = engine.execute_layers(
            input_tensor.data,
            input_tensor.shape,
            position_ids,
        ).await?;

        let output_tensor = TensorData::new(output_data, output_shape);
        output_tensor.validate()?;

        Ok(output_tensor)
    }
}
```

---

### **Step 3: GGUF Layer Loading** ⭐

**File**: `crates/q-ai-inference/src/gguf_loader.rs`

**What to Add**:

```rust
impl GGUFModelLoader {
    /// Load only specific layers from GGUF (for distributed inference)
    pub fn load_layers(&self, start_layer: usize, end_layer: usize) -> Result<Vec<MistralLayerWeights>> {
        let mut layer_weights = Vec::new();

        for layer_idx in start_layer..=end_layer {
            let weights = self.load_single_layer(layer_idx)?;
            layer_weights.push(weights);
        }

        Ok(layer_weights)
    }

    /// Load embedding layer (for first node)
    pub fn load_embedding(&self) -> Result<Tensor> {
        // Load "token_embd.weight" from GGUF
        let embedding_tensor = self.get_tensor("token_embd.weight")?;
        Ok(embedding_tensor)
    }

    /// Load LM head (for last node)
    pub fn load_lm_head(&self) -> Result<Tensor> {
        // Load "output.weight" from GGUF
        let lm_head_tensor = self.get_tensor("output.weight")?;
        Ok(lm_head_tensor)
    }
}
```

---

### **Step 4: Tensor Serialization for Network** ✅

**File**: `crates/q-network/src/layer_forwarding.rs`

**Status**: Already implemented!
- ✅ `TensorData::serialize()` - To bytes
- ✅ `TensorData::deserialize()` - From bytes
- ✅ Sent via gossipsub as `AIMessagePayload::LayerOutput`

---

### **Step 5: Autoregressive Token Generation** 🎯

**File**: `crates/q-network/src/distributed_ai_coordinator.rs`

**Add Generation Loop**:

```rust
pub async fn generate_distributed(
    &self,
    prompt: &str,
    max_tokens: usize,
    model: &str,
) -> Result<String> {
    let mut generated_text = String::new();
    let mut current_prompt = prompt.to_string();

    for token_idx in 0..max_tokens {
        // Forward pass through pipeline
        let logits = self.forward_pass(&current_prompt, model).await?;

        // Sample next token (on last node)
        let next_token_id = self.sample_token(&logits, 0.7)?;

        // Decode token
        let next_token_text = self.decode_token(next_token_id)?;
        generated_text.push_str(&next_token_text);

        // Append to prompt for next iteration
        current_prompt = format!("{}{}", prompt, &generated_text);

        // Stop on EOS token
        if next_token_id == 2 { // EOS token ID
            break;
        }
    }

    Ok(generated_text)
}
```

---

## 📊 **Implementation Checklist**

### **Core Engine**:
- [ ] 1. Implement `DistributedMistralEngine::load_from_gguf()` with layer range
- [ ] 2. Implement `DistributedMistralEngine::execute_layers()` using real MistralLayer
- [ ] 3. Implement `DistributedMistralEngine::get_embeddings()` for first node
- [ ] 4. Add GGUF selective layer loading: `load_layers(start, end)`
- [ ] 5. Add GGUF embedding loading: `load_embedding()`
- [ ] 6. Add GGUF LM head loading: `load_lm_head()`

### **Worker Integration**:
- [ ] 7. Add `engine: Option<Arc<DistributedMistralEngine>>` to worker
- [ ] 8. Implement `initialize_engine()` in worker
- [ ] 9. Replace placeholder `run_model_layers()` with real execution
- [ ] 10. Test single-node layer execution (layers 0-7)

### **Coordinator Integration**:
- [ ] 11. Implement `generate_distributed()` with autoregressive loop
- [ ] 12. Add token sampling on last node
- [ ] 13. Add KV-cache coordination (optional for v1)

### **Testing**:
- [ ] 14. Test 2-node pipeline (layers 0-15, 16-31)
- [ ] 15. Test 4-node pipeline (layers 0-7, 8-15, 16-23, 24-31)
- [ ] 16. Measure latency improvement vs single node
- [ ] 17. Test with actual Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

---

## 🎯 **Performance Targets**

### **Single Node (Baseline)**:
- Mistral-7B: 32 layers sequentially
- Latency per token: ~2s on CPU

### **4-Node Pipeline**:
- Node 1: Layers 0-7 (0.5s)
- Node 2: Layers 8-15 (0.5s)
- Node 3: Layers 16-23 (0.5s)
- Node 4: Layers 24-31 (0.5s)
- **Pipeline latency per token: ~0.5s** (4× faster!) ⚡

**After pipeline fills**: All 4 nodes work simultaneously
- Token 1: Node 4 (final layers)
- Token 2: Node 3 (middle layers)
- Token 3: Node 2 (middle layers)
- Token 4: Node 1 (first layers)

**Throughput = 4× after pipeline warmup!**

---

## 🚀 **Advantages**

| Feature | Single Node | 4-Node Pipeline |
|---------|------------|----------------|
| **Memory per Node** | 4.3 GB (full model) | ~1.1 GB (8 layers) |
| **Latency per Token** | 2s | **0.5s** (4× faster) |
| **Can Run Mistral-Small-3.2-24B** | ❌ (14 GB) | ✅ (3.5 GB per node) |
| **Complexity** | Low | Medium |

---

## ⏱️ **Timeline**

**Estimated Time**: 6-8 hours

- [ ] **Phase 1**: DistributedMistralEngine implementation (3-4 hours)
- [ ] **Phase 2**: GGUF selective loading (1-2 hours)
- [ ] **Phase 3**: Worker integration (1 hour)
- [ ] **Phase 4**: Coordinator generation loop (1 hour)
- [ ] **Phase 5**: Testing and optimization (1 hour)

---

## 🎉 **Success Criteria**

- ✅ Node loads only assigned 8 layers (1.1 GB memory)
- ✅ Forward pass executes through real MistralLayer
- ✅ Hidden states transfer between nodes via gossipsub
- ✅ 4-node pipeline generates tokens 4× faster
- ✅ Can run Mistral-Small-3.2-24B distributed (impossible on single node)

---

**Status**: Ready to implement TRUE pipeline parallelism! 🚀
