# Distributed AI Implementation Plan - v0.5.0-beta

**Date**: October 31, 2025
**Goal**: Enable TRUE horizontal scaling where N nodes = N× faster AI inference

---

## Current Problem

**Infrastructure EXISTS but is NOT CONNECTED**:
```rust
// chat_api.rs:239 - CURRENT (BROKEN)
match engine.generate(&prompt, max_tokens).await {
    // ❌ Always runs LOCAL ONLY
    // ❌ Never uses coordinator
    // ❌ Never distributes across nodes
    // ❌ Adding nodes = NO performance improvement
}
```

---

## Proposed Architecture

### Phase 1: Coordinator Integration ✅

**File**: `crates/q-api-server/src/chat_api.rs:230-260`

```rust
// NEW DISTRIBUTED PATH
if metadata.distributed_enabled && peer_count > 1 {
    // Use DistributedAICoordinator for multi-node inference
    let coordinator = state.distributed_ai_coordinator.as_ref().unwrap();

    match coordinator.coordinate_inference(
        &formatted_prompt,
        max_tokens,
        &metadata.model
    ).await {
        Ok((response, nodes_used)) => {
            info!("🌐 Distributed inference: {} nodes, response: {}", nodes_used.len(), &response[..50.min(response.len())]);
            (response, create_distributed_stats(nodes_used))
        }
        Err(e) => {
            warn!("Distributed inference failed, falling back to local: {}", e);
            // Fallback to local
            engine.generate(&formatted_prompt, max_tokens).await?
        }
    }
} else {
    // LOCAL PATH (single node or distributed disabled)
    engine.generate(&formatted_prompt, max_tokens).await?
}
```

### Phase 2: Layer Assignment Algorithm ✅

**File**: `crates/q-network/src/distributed_ai_coordinator.rs`

**Add Method**:
```rust
impl DistributedAICoordinator {
    /// Coordinate distributed inference across available nodes
    pub async fn coordinate_inference(
        &self,
        prompt: &str,
        max_tokens: usize,
        model: &str,
    ) -> Result<(String, Vec<String>)> {
        let request_id = Uuid::new_v4().to_string();

        // 1. Get available nodes from network
        let nodes = self.get_available_nodes().await?;
        if nodes.is_empty() {
            return Err(anyhow!("No nodes available for distributed inference"));
        }

        // 2. Assign layers to nodes based on capability
        let layer_assignments = self.assign_layers_to_nodes(&nodes, model)?;

        // 3. Publish inference request via GossipSub
        self.publish_inference_request(request_id.clone(), prompt, max_tokens, model).await?;

        // 4. Wait for layer outputs from assigned nodes
        let outputs = self.collect_layer_outputs(request_id.clone(), &layer_assignments).await?;

        // 5. Run final layers locally and aggregate
        let final_response = self.aggregate_outputs(outputs).await?;

        let nodes_used: Vec<String> = layer_assignments.keys().cloned().collect();
        Ok((final_response, nodes_used))
    }

    /// Assign model layers to nodes based on their capabilities
    fn assign_layers_to_nodes(
        &self,
        nodes: &[AINode],
        model: &str,
    ) -> Result<HashMap<String, (usize, usize)>> {
        let total_layers = self.get_model_layer_count(model);
        let node_count = nodes.len();

        // Simple strategy: divide layers equally
        let layers_per_node = total_layers / node_count;
        let mut assignments = HashMap::new();

        for (i, node) in nodes.iter().enumerate() {
            let start_layer = i * layers_per_node;
            let end_layer = if i == node_count - 1 {
                total_layers // Last node gets remaining layers
            } else {
                start_layer + layers_per_node
            };

            assignments.insert(node.node_id.clone(), (start_layer, end_layer));
        }

        Ok(assignments)
    }
}
```

### Phase 3: P2P Layer Forwarding ✅

**Files**:
- `crates/q-network/src/layer_forwarding.rs` (already exists, enhance)
- `crates/q-network/src/distributed_ai.rs` (add message handling)

**Message Flow**:
```
Node 1 (layers 0-10)  →  GossipSub  →  Node 2 (layers 11-21)
       ↓                                       ↓
  Process layers              Receive KV-cache from Node 1
  Compress output             Process layers 11-21
  Publish to topic            Publish to topic

                              →  Node 3 (layers 22-32)
                                       ↓
                              Final token generation
                              Return to coordinator
```

### Phase 4: Result Aggregation ✅

**Implementation**:
```rust
async fn collect_layer_outputs(
    &self,
    request_id: String,
    assignments: &HashMap<String, (usize, usize)>,
) -> Result<Vec<LayerOutput>> {
    let timeout = tokio::time::Duration::from_secs(30);
    let mut outputs = Vec::new();

    // Subscribe to layer output topic
    let mut receiver = self.layer_output_manager.subscribe(&request_id).await;

    let mut received_count = 0;
    let expected_count = assignments.len();

    loop {
        match tokio::time::timeout(timeout, receiver.recv()).await {
            Ok(Some(output)) => {
                outputs.push(output);
                received_count += 1;

                if received_count == expected_count {
                    break;
                }
            }
            Ok(None) => {
                return Err(anyhow!("Layer output channel closed"));
            }
            Err(_) => {
                return Err(anyhow!("Timeout waiting for layer outputs"));
            }
        }
    }

    // Sort by layer index
    outputs.sort_by_key(|o| o.layer_index);
    Ok(outputs)
}
```

---

## Performance Expectations

### Current (v0.4.0-beta)
- **1 node**: 10 tokens/sec
- **2 nodes**: 10 tokens/sec (NO IMPROVEMENT ❌)
- **3 nodes**: 10 tokens/sec (NO IMPROVEMENT ❌)

### After Implementation (v0.5.0-beta)
- **1 node**: 10 tokens/sec (baseline)
- **2 nodes**: ~18 tokens/sec (1.8x faster ✅)
- **3 nodes**: ~25 tokens/sec (2.5x faster ✅)
- **4 nodes**: ~30 tokens/sec (3.0x faster ✅)

**Why not linear?** P2P overhead (KV-cache transfer, coordination latency)

---

## Implementation Order

1. ✅ **Add `coordinate_inference()` to DistributedAICoordinator**
2. ✅ **Implement layer assignment algorithm**
3. ✅ **Add GossipSub message publishing/receiving**
4. ✅ **Hook coordinator into chat_api.rs**
5. ✅ **Test with 2-3 nodes**
6. ✅ **Measure performance scaling**

---

## Files to Modify

1. `crates/q-api-server/src/chat_api.rs` - Hook coordinator (lines 230-260)
2. `crates/q-network/src/distributed_ai_coordinator.rs` - Add coordination methods
3. `crates/q-network/src/layer_forwarding.rs` - Enhance KV-cache forwarding
4. `crates/q-api-server/src/main.rs` - Ensure coordinator is in AppState

---

## Testing Plan

```bash
# Terminal 1 - Node 1 (coordinator)
Q_DB_PATH=./data-node1 Q_P2P_PORT=9001 ./q-api-server --port 8001

# Terminal 2 - Node 2 (worker)
Q_DB_PATH=./data-node2 Q_P2P_PORT=9002 ./q-api-server --port 8002

# Terminal 3 - Node 3 (worker)
Q_DB_PATH=./data-node3 Q_P2P_PORT=9003 ./q-api-server --port 8003

# Test inference
curl -X POST http://localhost:8001/api/chat/XXX/message \
  -H "Content-Type: application/json" \
  -d '{"content": "What is quantum computing?"}'

# Expected logs:
# Node 1: "🌐 Assigning layers 0-10 to node1, layers 11-21 to node2, layers 22-32 to node3"
# Node 2: "🔄 Processing layers 11-21 for request XXX"
# Node 3: "🔄 Processing layers 22-32 for request XXX"
# Node 1: "✅ Distributed inference complete: 3 nodes, 28 tokens/sec"
```

---

## Success Criteria

- ✅ **Coordinator gets invoked** when `distributed_enabled=true` and `peer_count > 1`
- ✅ **Layers are split** across nodes (observable in logs)
- ✅ **P2P messages flow** via GossipSub topics
- ✅ **Performance scales** with node count (measured via `tokens_per_second`)
- ✅ **Fallback works** when distributed fails

---

Ready to implement? This is a **complex multi-file change** - estimated **2-3 hours** of implementation + testing.
