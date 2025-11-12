# Distributed AI: Horizontal Scaling Implementation Plan

## 🎯 **Goal: N Nodes = N× Throughput**

**Date**: November 6, 2025
**Version**: v0.9.27-beta
**Status**: Implementation Phase

---

## 📊 **Revised Architecture: Practical Horizontal Scaling**

### **Why Not Pipeline Parallelism?**

**Original Plan** (Pipeline Parallelism):
```
Node 1: Layers 0-7  → Node 2: Layers 8-15 → Node 3: Layers 16-23 → Node 4: Layers 24-31
```

**Problem**:
- Requires layer-by-layer model execution
- mistralrs_core doesn't expose layer APIs publicly
- DistributedMistralEngine is a stub awaiting public APIs
- Would need to modify mistral.rs internals

**Revised Plan** (Horizontal Scaling + Load Balancing):
```
                    ┌─────────────────────────────┐
                    │   Coordinator (Gateway)     │
                    │   - Route requests          │
                    │   - Aggregate responses     │
                    │   - Token streaming         │
                    └──────────┬──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐           ┌────▼────┐           ┌────▼────┐
   │ Worker 1│           │ Worker 2│           │ Worker 3│
   │ Full    │           │ Full    │           │ Full    │
   │ Mistral │           │ Mistral │           │ Mistral │
   │ Model   │           │ Model   │           │ Model   │
   └─────────┘           └─────────┘           └─────────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                     3 Concurrent Requests = 3× Throughput
```

**Advantages**:
✅ Uses MistralRsEngine (working public API)
✅ N nodes = N× concurrent request throughput
✅ Token streaming works out of the box
✅ Fault tolerance (if one node fails, others continue)
✅ Same architecture as production LLM services (OpenAI, Anthropic)

---

## 🏗️ **Implementation Steps**

### **Phase 1: Worker Node with MistralRsEngine** ✅ (Partially Complete)

**File**: `crates/q-network/src/distributed_ai_worker.rs`

**Current Status**:
- ✅ Worker struct exists
- ✅ Gossipsub message handling
- ❌ Placeholder inference implementation

**What to Implement**:

1. **Add MistralRsEngine to Worker**:
```rust
pub struct DistributedAIWorker {
    coordinator: Arc<DistributedAICoordinator>,
    active_requests: Arc<RwLock<HashMap<String, ActiveInferenceRequest>>>,

    // NEW: Add MistralRsEngine for actual inference
    inference_engine: Arc<tokio::sync::Mutex<q_ai_inference::MistralRsEngine>>,
    // OR use ModelManager for lazy loading
    model_manager: Arc<q_ai_inference::ModelManager>,
}
```

2. **Replace `run_model_layers` with Complete Inference**:
```rust
async fn execute_full_inference(
    &self,
    request_id: String,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    model: String,
) -> Result<()> {
    info!("🤖 Worker executing full inference: request={}", request_id);

    // Get or load model
    let engine = self.model_manager
        .get_or_load_model(&model)
        .await?;

    // Generate response with streaming
    let stream = engine.generate_stream(&prompt, max_tokens).await?;

    // Forward tokens back to coordinator via gossipsub
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Token(token) => {
                self.send_token_to_coordinator(&request_id, token).await?;
            }
            StreamEvent::Complete(stats) => {
                self.send_completion_to_coordinator(&request_id, stats).await?;
            }
            StreamEvent::Error(err) => {
                self.send_error_to_coordinator(&request_id, err).await?;
            }
        }
    }

    Ok(())
}
```

3. **Token Streaming via Gossipsub**:
```rust
async fn send_token_to_coordinator(&self, request_id: &str, token: String) -> Result<()> {
    let message = AIGossipsubMessage {
        message_id: uuid::Uuid::new_v4().to_string(),
        sender_node_id: self.coordinator.node_id.clone(),
        sender_peer_id: self.coordinator.peer_id.clone(),
        timestamp: chrono::Utc::now().timestamp(),
        payload: AIMessagePayload::InferenceToken {
            request_id: request_id.to_string(),
            token,
        },
    };

    self.coordinator.publish_message(message).await?;
    Ok(())
}
```

---

### **Phase 2: Coordinator Load Balancing** ⚠️ (Needs Update)

**File**: `crates/q-network/src/distributed_ai_coordinator.rs`

**Current Status**:
- ✅ Coordinator struct exists
- ✅ Node discovery and heartbeats
- ⚠️ coordinate_inference assumes pipeline parallelism

**What to Update**:

1. **Simplify `coordinate_inference` for Round-Robin Load Balancing**:
```rust
pub async fn coordinate_inference(
    &self,
    prompt: &str,
    max_tokens: usize,
    model: &str,
) -> Result<(String, Vec<String>)> {
    let request_id = uuid::Uuid::new_v4().to_string();

    // 1. Get available worker nodes
    let nodes = self.get_available_nodes().await?;
    if nodes.is_empty() {
        return Err(anyhow!("No worker nodes available"));
    }

    // 2. Select least-loaded node (simple round-robin for now)
    let selected_node = self.select_worker_node(&nodes).await?;
    info!("🎯 Routing request {} to worker {}", request_id, selected_node.node_id);

    // 3. Publish inference request to selected worker
    self.publish_worker_request(
        request_id.clone(),
        selected_node.node_id.clone(),
        prompt,
        max_tokens,
        model,
    ).await?;

    // 4. Stream tokens back from worker
    let response_rx = self.register_response_channel(&request_id).await?;
    let mut full_response = String::new();

    while let Some(chunk) = response_rx.recv().await {
        match chunk {
            InferenceResponseChunk::Token(token) => {
                full_response.push_str(&token);
            }
            InferenceResponseChunk::Complete { .. } => {
                break;
            }
            InferenceResponseChunk::Error(err) => {
                return Err(anyhow!("Worker error: {}", err));
            }
        }
    }

    Ok((full_response, vec![selected_node.node_id]))
}
```

2. **Add Load Balancing Logic**:
```rust
async fn select_worker_node(&self, nodes: &[AINode]) -> Result<AINode> {
    // Strategy 1: Least active requests
    let selected = nodes
        .iter()
        .min_by_key(|n| n.active_requests)
        .ok_or_else(|| anyhow!("No nodes available"))?
        .clone();

    Ok(selected)
}
```

---

### **Phase 3: Gossipsub Message Updates** ✅ (Exists, Needs Token Message)

**File**: `crates/q-network/src/distributed_ai.rs`

**What to Add**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIMessagePayload {
    // Existing messages...
    Heartbeat { ... },
    InferenceRequest { ... },
    LayerAssignment { ... },

    // NEW: Token streaming messages
    InferenceToken {
        request_id: String,
        token: String,
    },
    InferenceComplete {
        request_id: String,
        total_tokens: usize,
        latency_ms: u64,
    },
    InferenceError {
        request_id: String,
        error: String,
    },
}
```

---

### **Phase 4: API Integration** ✅ (Already Works!)

**File**: `crates/q-api-server/src/chat_api.rs`

**Current Implementation** (lines 254-287):
```rust
if metadata.distributed_enabled && peer_count > 0 {
    match coordinator.coordinate_inference(&formatted_prompt, max_tokens, &metadata.model).await {
        Ok((response, nodes_used)) => {
            // Distributed inference successful
            (response, GenerationStats { ... })
        }
        Err(e) => {
            // Falls back to local MistralRsEngine
            warn!("Distributed inference failed: {}", e);
        }
    }
}
```

**Status**: ✅ **No changes needed!** The API layer already:
- Calls `coordinator.coordinate_inference()`
- Falls back to local inference on failure
- Returns response to frontend

---

## 📋 **Implementation Checklist**

### **Core Components**:
- [ ] 1. Add `InferenceToken`, `InferenceComplete`, `InferenceError` to `AIMessagePayload`
- [ ] 2. Add `MistralRsEngine` or `ModelManager` to `DistributedAIWorker`
- [ ] 3. Implement `execute_full_inference` in worker (replaces layer-based execution)
- [ ] 4. Implement `send_token_to_coordinator` in worker
- [ ] 5. Update `coordinate_inference` to use load balancing (not pipeline)
- [ ] 6. Implement `select_worker_node` for load balancing
- [ ] 7. Handle `InferenceToken` messages in coordinator's message handler
- [ ] 8. Stream tokens from coordinator to API layer

### **Testing**:
- [ ] 9. Test single-node distributed inference
- [ ] 10. Test 2-node concurrent requests (2× throughput)
- [ ] 11. Test 3-node concurrent requests (3× throughput)
- [ ] 12. Test worker failure and recovery
- [ ] 13. Benchmark throughput scaling (1 node vs N nodes)

### **Deployment**:
- [ ] 14. Build release binary: `timeout 36000 cargo build --release`
- [ ] 15. Deploy to Server Beta (185.182.185.227)
- [ ] 16. Test distributed AI from production frontend

---

## 🎯 **Performance Targets**

### **Single Node Baseline**:
- Mistral-7B-Instruct-v0.3: ~15 tokens/s on CPU, ~50 tokens/s on GPU
- Latency per request: ~10-30s for 150 tokens

### **Distributed (3 Nodes)**:
- **Concurrent Throughput**: 3× baseline (3 requests simultaneously)
- **Individual Latency**: Same as single node (~10-30s per request)
- **Total Throughput**: 45-150 tokens/s across all requests

### **Key Metrics**:
```
1 Node:  1 request  = 15 tok/s → User waits 10s
3 Nodes: 3 requests = 45 tok/s → Each user waits 10s (but 3× more users served)
```

**This achieves N nodes = N× throughput!** ✅

---

## 🚀 **Advantages Over Pipeline Parallelism**

| Feature | Pipeline Parallelism | Horizontal Scaling |
|---------|---------------------|-------------------|
| **Implementation** | Requires private APIs | Uses public MistralRsEngine |
| **Complexity** | Very High (layer forwarding, KV-cache sync) | Medium (load balancing) |
| **Latency per Request** | Lower (N nodes process 1 request faster) | Same as single node |
| **Throughput** | 1× (only 1 request at a time) | **N× (N concurrent requests)** |
| **Fault Tolerance** | Low (if one node fails, request fails) | High (failed node doesn't affect others) |
| **Production Readiness** | Experimental | Industry standard ✅ |

**For most use cases, horizontal scaling is BETTER** because:
1. Users care more about "can I get a response now" than "is this 2s faster"
2. N× throughput means N× more users can be served simultaneously
3. Much simpler to implement and maintain

---

## 📊 **Timeline**

**Estimated Time**: 4-6 hours

- [ ] **Phase 1**: Message types + Worker engine integration (1-2 hours)
- [ ] **Phase 2**: Coordinator load balancing update (1 hour)
- [ ] **Phase 3**: Token streaming implementation (1-2 hours)
- [ ] **Phase 4**: Testing with multiple nodes (1 hour)
- [ ] **Phase 5**: Build and deployment (30min)

---

## 🎉 **Success Criteria**

- ✅ Worker nodes can execute full inference with MistralRsEngine
- ✅ Coordinator routes requests to least-loaded worker
- ✅ Tokens stream from worker → coordinator → API → frontend
- ✅ 3 nodes handle 3 concurrent requests simultaneously (3× throughput)
- ✅ Frontend distributed AI chat works end-to-end

---

**Status**: Ready to implement! 🚀
