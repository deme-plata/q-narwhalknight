# Q-NarwhalKnight Distributed AI - Phase 2 Roadmap

**Date:** October 28, 2025
**Current Status:** Phase 1 Complete ✅
**Next Phase:** Forward Pass & Distributed Inference

---

## 📊 Current Achievement Summary

### ✅ Phase 1: Foundation (COMPLETED)

**What's Working:**
- ✅ Real GGUF tokenizer extraction (32,768 vocab, 736ms)
- ✅ Production sampling strategies (temperature, top-k, top-p)
- ✅ Autoregressive generation loop (with stop conditions)
- ✅ GGUF model loading (4.1GB Mistral-7B)
- ✅ Special layers loaded (embeddings, norm, output)
- ✅ 291 tensors accessible from GGUF
- ✅ 8,068 lines of production code

**Files Created:**
```
crates/q-ai-inference/src/
├── gguf_tokenizer.rs    (451 lines) - Real GGUF tokenizer
├── sampling.rs          (379 lines) - Sampling strategies
├── generation.rs        (389 lines) - Generation loop
├── tokenizer.rs         (updated)   - Integration
└── lib.rs               (updated)   - Module exports

crates/q-ai-inference/examples/
└── test_real_inference.rs (163 lines) - End-to-end validation
```

---

## 🎯 Phase 2: Forward Pass & Distributed Inference

### Goal
Implement the actual neural network forward pass through Mistral layers and enable distributed computation across the libp2p network.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                  Distributed Inference Flow                │
└────────────────────────────────────────────────────────────┘

User Prompt
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  Encode prompt → tokens
│   (GGUF)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │  tokens → hidden_states [1, seq_len, 4096]
│  Layer          │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Distributed Layer Processing               │
│                                                          │
│  Node A: Layers 0-10  →  Node B: Layers 11-21  →       │
│                                                          │
│  Node C: Layers 22-31                                   │
│                                                          │
│  (Tensor communication via libp2p)                      │
└────────────────────────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Norm +   │  hidden → normalized → logits [1, seq_len, 32768]
│  Output Proj    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Sampling      │  logits → next_token
│   Strategy      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Detokenizer    │  token → text
└─────────────────┘
```

---

## 📋 Phase 2 Tasks

### Task 1: Implement Mistral Layer Forward Pass ⏳

**Priority:** HIGH
**Complexity:** Medium
**Estimated Time:** 3-4 hours

**Implementation Steps:**

1. **Complete MistralLayer Implementation**
   - File: `crates/q-ai-inference/src/mistral_model.rs`
   - Current: Stub implementation exists
   - Need: Real forward pass using loaded weights

2. **Components to Implement:**
   ```rust
   pub struct MistralLayer {
       // Attention
       self_attn: MistralAttention,

       // Feed-forward network
       mlp: MistralMLP,

       // Layer norms
       input_layernorm: RMSNorm,
       post_attention_layernorm: RMSNorm,
   }

   impl MistralLayer {
       pub fn forward(
           &self,
           hidden_states: &Tensor,
           attention_mask: Option<&Tensor>,
           position_ids: Option<&Tensor>,
           kv_cache: Option<&mut KVCache>,
       ) -> Result<Tensor> {
           // 1. Input layer norm
           let normed = self.input_layernorm.forward(hidden_states)?;

           // 2. Self-attention
           let attn_output = self.self_attn.forward(
               &normed,
               attention_mask,
               position_ids,
               kv_cache
           )?;

           // 3. Residual connection
           let hidden_states = (hidden_states + attn_output)?;

           // 4. Post-attention norm
           let normed = self.post_attention_layernorm.forward(&hidden_states)?;

           // 5. MLP
           let mlp_output = self.mlp.forward(&normed)?;

           // 6. Residual connection
           let output = (hidden_states + mlp_output)?;

           Ok(output)
       }
   }
   ```

3. **Attention Mechanism:**
   ```rust
   pub struct MistralAttention {
       q_proj: Linear,
       k_proj: Linear,
       v_proj: Linear,
       o_proj: Linear,

       num_heads: usize,
       num_kv_heads: usize,  // 8 for GQA
       head_dim: usize,

       rotary_emb: RotaryEmbedding,
   }

   impl MistralAttention {
       pub fn forward(
           &self,
           hidden_states: &Tensor,
           attention_mask: Option<&Tensor>,
           position_ids: Option<&Tensor>,
           kv_cache: Option<&mut KVCache>,
       ) -> Result<Tensor> {
           // 1. Project to Q, K, V
           let query = self.q_proj.forward(hidden_states)?;
           let key = self.k_proj.forward(hidden_states)?;
           let value = self.v_proj.forward(hidden_states)?;

           // 2. Reshape for multi-head attention
           // [batch, seq_len, hidden] → [batch, num_heads, seq_len, head_dim]

           // 3. Apply rotary positional embeddings
           let (query, key) = self.rotary_emb.apply(&query, &key, position_ids)?;

           // 4. Grouped-query attention (GQA)
           // Repeat K, V for GQA: 8 KV heads → 32 Q heads

           // 5. Update KV cache if provided
           if let Some(cache) = kv_cache {
               cache.update(&key, &value)?;
           }

           // 6. Scaled dot-product attention
           let attn_weights = query.matmul(&key.transpose(-2, -1)?)?;
           let attn_weights = (attn_weights / (self.head_dim as f64).sqrt())?;

           // 7. Apply attention mask if provided
           if let Some(mask) = attention_mask {
               let attn_weights = attn_weights.broadcast_add(mask)?;
           }

           // 8. Softmax
           let attn_weights = candle_nn::ops::softmax(&attn_weights, -1)?;

           // 9. Apply attention to values
           let attn_output = attn_weights.matmul(&value)?;

           // 10. Reshape back and project
           // [batch, num_heads, seq_len, head_dim] → [batch, seq_len, hidden]
           let output = self.o_proj.forward(&attn_output)?;

           Ok(output)
       }
   }
   ```

4. **Feed-Forward Network:**
   ```rust
   pub struct MistralMLP {
       gate_proj: Linear,  // Linear(4096, 14336)
       up_proj: Linear,    // Linear(4096, 14336)
       down_proj: Linear,  // Linear(14336, 4096)
   }

   impl MistralMLP {
       pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
           // SwiGLU activation
           let gate = self.gate_proj.forward(hidden_states)?;
           let gate = silu(&gate)?;  // SiLU activation

           let up = self.up_proj.forward(hidden_states)?;
           let hidden = (gate * up)?;

           let output = self.down_proj.forward(&hidden)?;
           Ok(output)
       }
   }
   ```

**Validation:**
```rust
#[test]
fn test_mistral_layer_forward() {
    let config = MistralConfig::default();
    let layer = MistralLayer::from_weights(&weights, &config, &device)?;

    let input = Tensor::randn(0.0, 1.0, &[1, 10, 4096], &device)?;
    let output = layer.forward(&input, None, None, None)?;

    assert_eq!(output.shape(), &[1, 10, 4096]);
}
```

### Task 2: KV-Cache Implementation ⏳

**Priority:** HIGH
**Complexity:** Medium
**Estimated Time:** 2-3 hours

**Purpose:** Cache key/value tensors to avoid recomputation during autoregressive generation.

**Implementation:**
```rust
pub struct KVCache {
    key_cache: Vec<Tensor>,    // One per layer
    value_cache: Vec<Tensor>,  // One per layer
    current_length: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, config: &MistralConfig) -> Result<Self> {
        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            // Pre-allocate cache tensors
            let key = Tensor::zeros(
                &[1, config.num_kv_heads, max_seq_len, config.head_dim],
                DType::F32,
                &device
            )?;
            let value = Tensor::zeros(
                &[1, config.num_kv_heads, max_seq_len, config.head_dim],
                DType::F32,
                &device
            )?;

            key_cache.push(key);
            value_cache.push(value);
        }

        Ok(Self {
            key_cache,
            value_cache,
            current_length: 0,
        })
    }

    pub fn update(&mut self, layer_idx: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        // Append new key/value to cache
        // This allows incremental generation without recomputing past tokens

        let cache_key = &mut self.key_cache[layer_idx];
        let cache_value = &mut self.value_cache[layer_idx];

        // Update cache at current position
        // Implementation details depend on Candle's tensor update API

        Ok(())
    }

    pub fn get(&self, layer_idx: usize) -> (&Tensor, &Tensor) {
        (&self.key_cache[layer_idx], &self.value_cache[layer_idx])
    }
}
```

**Performance Impact:**
- **Without KV-cache:** O(n²) for each new token
- **With KV-cache:** O(n) for each new token
- **Expected speedup:** 3-5x for typical generation

### Task 3: Distributed Layer Execution ⏳

**Priority:** HIGH
**Complexity:** High
**Estimated Time:** 4-6 hours

**Architecture:**

```
Node Assignment Example (32 layers, 3 nodes):

Node A (GPU, 16GB VRAM):
  - Layers 0-15 (16 layers)
  - Estimated load time: ~4s
  - Inference time: ~50ms per token

Node B (GPU, 8GB VRAM):
  - Layers 16-23 (8 layers)
  - Estimated load time: ~2s
  - Inference time: ~25ms per token

Node C (CPU, 32GB RAM):
  - Layers 24-31 (8 layers)
  - Estimated load time: ~2s
  - Inference time: ~100ms per token

Total inference time per token: ~175ms (pipeline parallel)
```

**Implementation:**

1. **Layer Assignment Protocol:**
```rust
pub struct LayerAssignment {
    pub node_id: String,
    pub peer_id: PeerId,
    pub layer_start: usize,
    pub layer_end: usize,
    pub device_capability: DeviceCapability,
    pub last_seen: i64,
}

pub struct DistributedInferenceCoordinator {
    layer_assignments: Arc<RwLock<HashMap<String, LayerAssignment>>>,
    gossipsub: Gossipsub,
}

impl DistributedInferenceCoordinator {
    pub async fn assign_layers(&self) -> Result<Vec<LayerAssignment>> {
        // 1. Discover available nodes via gossipsub
        let peers = self.discover_peers().await?;

        // 2. Query capabilities from each peer
        let capabilities = self.query_capabilities(&peers).await?;

        // 3. Optimize layer assignment based on:
        //    - Device capability (CUDA > Metal > CPU)
        //    - Network latency
        //    - Current load
        let assignments = self.optimize_assignment(capabilities)?;

        // 4. Broadcast assignments via gossipsub
        self.broadcast_assignments(&assignments).await?;

        Ok(assignments)
    }
}
```

2. **Tensor Communication:**
```rust
#[derive(Serialize, Deserialize)]
pub struct TensorMessage {
    request_id: String,
    layer_index: usize,
    tensor_data: Vec<u8>,  // Compressed
    shape: Vec<usize>,
    dtype: String,
}

impl TensorMessage {
    pub fn compress(tensor: &Tensor) -> Result<Self> {
        // 1. Convert tensor to bytes
        let data = tensor.to_vec1::<f32>()?;
        let bytes = bincode::serialize(&data)?;

        // 2. Compress with flate2
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast()
        );
        encoder.write_all(&bytes)?;
        let compressed = encoder.finish()?;

        Ok(Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            layer_index: 0,
            tensor_data: compressed,
            shape: tensor.dims().to_vec(),
            dtype: "f32".to_string(),
        })
    }

    pub fn decompress(&self, device: &Device) -> Result<Tensor> {
        // 1. Decompress
        let mut decoder = flate2::read::GzDecoder::new(&self.tensor_data[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;

        // 2. Deserialize
        let data: Vec<f32> = bincode::deserialize(&decompressed)?;

        // 3. Create tensor
        let tensor = Tensor::from_vec(data, &self.shape, device)?;

        Ok(tensor)
    }
}
```

3. **Distributed Forward Pass:**
```rust
pub async fn distributed_forward(
    &self,
    input_tokens: &[u32],
    assignments: &[LayerAssignment],
) -> Result<String> {
    // 1. Tokenize locally
    let tokens = input_tokens.to_vec();

    // 2. Get embeddings locally
    let mut hidden_states = self.embed_tokens(&tokens)?;

    // 3. Process layers across nodes
    for assignment in assignments {
        if assignment.node_id == self.local_node_id {
            // Local processing
            hidden_states = self.process_layers_local(
                hidden_states,
                assignment.layer_start,
                assignment.layer_end
            )?;
        } else {
            // Remote processing
            hidden_states = self.process_layers_remote(
                hidden_states,
                assignment
            ).await?;
        }
    }

    // 4. Final norm + output projection
    let logits = self.final_forward(&hidden_states)?;

    // 5. Sample next token
    let next_token = self.sample(&logits)?;

    // 6. Decode
    let text = self.tokenizer.decode(&[next_token], true)?;

    Ok(text)
}

async fn process_layers_remote(
    &self,
    hidden_states: Tensor,
    assignment: &LayerAssignment,
) -> Result<Tensor> {
    // 1. Compress tensor
    let msg = TensorMessage::compress(&hidden_states)?;

    // 2. Send via gossipsub
    let topic = format!("/qnk/inference/{}", assignment.node_id);
    self.gossipsub.publish(topic, bincode::serialize(&msg)?)?;

    // 3. Wait for response (with timeout)
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        self.wait_for_response(&msg.request_id)
    ).await??;

    // 4. Decompress response
    let output = response.decompress(&self.device)?;

    Ok(output)
}
```

### Task 4: Privacy Layer Integration ⏳

**Priority:** MEDIUM
**Complexity:** High
**Estimated Time:** 6-8 hours

**Components:**

1. **Tensor Encryption (AEGIS-QL):**
```rust
pub struct PrivacyLayer {
    aegis: Arc<AegisQL>,
    zk_system: Arc<ZKStarkSystem>,
}

impl PrivacyLayer {
    pub fn encrypt_tensor(&self, tensor: &Tensor) -> Result<EncryptedTensor> {
        // 1. Convert tensor to bytes
        let data = tensor_to_bytes(tensor)?;

        // 2. Encrypt with AEGIS-QL
        let ciphertext = self.aegis.encrypt(&data)?;

        // 3. Generate ZK proof of correct encryption
        let proof = self.zk_system.prove_encryption(&data, &ciphertext)?;

        Ok(EncryptedTensor {
            ciphertext,
            proof,
            shape: tensor.dims().to_vec(),
        })
    }

    pub fn compute_on_encrypted(
        &self,
        encrypted: &EncryptedTensor,
        operation: TensorOp,
    ) -> Result<EncryptedTensor> {
        // Homomorphic computation on encrypted tensors
        // This is complex - may require approximations or secure MPC
        todo!("Implement homomorphic tensor operations")
    }
}
```

2. **Zero-Knowledge Proofs for Computation:**
```rust
pub struct ComputationProof {
    input_commitment: Vec<u8>,
    output_commitment: Vec<u8>,
    proof: ZKProof,
}

impl PrivacyLayer {
    pub fn prove_computation(
        &self,
        input: &Tensor,
        output: &Tensor,
        layer_weights_hash: &[u8],
    ) -> Result<ComputationProof> {
        // Generate ZK-STARK proof that:
        // output = layer.forward(input) with specific weights

        // This allows verification without revealing:
        // - Input values
        // - Weight values
        // - Intermediate computations

        let input_commit = self.zk_system.commit(&tensor_to_bytes(input)?)?;
        let output_commit = self.zk_system.commit(&tensor_to_bytes(output)?)?;

        let proof = self.zk_system.prove_computation(
            input,
            output,
            layer_weights_hash
        )?;

        Ok(ComputationProof {
            input_commitment: input_commit,
            output_commitment: output_commit,
            proof,
        })
    }
}
```

---

## 🎯 Success Metrics for Phase 2

| Component | Metric | Target | Priority |
|-----------|--------|--------|----------|
| Forward Pass | Compiles | ✅ Yes | HIGH |
| Forward Pass | Correct Output | ✅ Match PyTorch | HIGH |
| KV-Cache | Speed Improvement | 3-5x | HIGH |
| Distributed | 2-node inference | ✅ Working | HIGH |
| Distributed | 3+ node inference | ✅ Working | MEDIUM |
| Privacy | Encrypted tensors | ✅ Working | MEDIUM |
| Privacy | ZK proofs | ✅ Verifiable | LOW |
| Performance | Tokens/sec | 10-40 | HIGH |
| Latency | Per token | <300ms | HIGH |

---

## 📚 Resources Needed

### Code References
1. **Candle Examples:** Study `candle/candle-examples/examples/mistral/`
2. **mistral.rs:** Reference for attention implementation
3. **HuggingFace Transformers:** For architecture details

### Testing Data
1. **Validation Set:** Use Mistral test prompts
2. **Ground Truth:** Compare with mistral.rs outputs
3. **Performance Baseline:** Mistral.cpp benchmarks

---

## 🔄 Implementation Order

### Week 1: Core Forward Pass
1. Day 1-2: MistralAttention implementation
2. Day 3: MistralMLP and layer implementation
3. Day 4: Integration testing
4. Day 5: KV-cache implementation

### Week 2: Distributed Inference
1. Day 1-2: Layer assignment protocol
2. Day 3-4: Tensor communication
3. Day 5: Multi-node testing

### Week 3: Privacy & Optimization
1. Day 1-2: Privacy layer integration
2. Day 3-4: Performance optimization
3. Day 5: End-to-end validation

---

## ✅ Validation Plan

### Unit Tests
```rust
#[test]
fn test_attention_mechanism() { /* ... */ }

#[test]
fn test_mlp_forward() { /* ... */ }

#[test]
fn test_layer_forward() { /* ... */ }

#[test]
fn test_kv_cache_update() { /* ... */ }
```

### Integration Tests
```rust
#[tokio::test]
async fn test_distributed_inference_2_nodes() { /* ... */ }

#[tokio::test]
async fn test_privacy_layer() { /* ... */ }
```

### Benchmarks
```rust
#[bench]
fn bench_single_token_generation(b: &mut Bencher) { /* ... */ }

#[bench]
fn bench_kv_cache_speedup(b: &mut Bencher) { /* ... */ }
```

---

## 🎓 Next Steps

1. **Start with Task 1:** Implement MistralAttention
2. **Validate Incrementally:** Test each component
3. **Compare with Reference:** Match mistral.rs outputs
4. **Optimize Performance:** Profile and improve
5. **Document Everything:** Follow CLAUDE.md principles

---

**Status:** 📋 **ROADMAP READY** - Phase 2 implementation can begin
