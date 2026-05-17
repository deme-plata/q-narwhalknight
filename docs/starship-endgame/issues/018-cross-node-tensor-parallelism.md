# Issue #018: Cross-Node Tensor Parallelism for Large Model Inference

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `ai-inference`, `p2p`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Large AI models (70B+ parameters) don't fit in a single node's VRAM/RAM. Cross-node tensor parallelism splits model layers across multiple peers connected via compute tunnels, enabling the network to serve models that no single node can run alone.

## Architecture

```
User Request: "Summarize this document" (needs 70B model)

Node A (16GB RAM)          Node B (24GB RAM)          Node C (16GB RAM)
  Layers 0-23                Layers 24-47               Layers 48-69
  ↓                          ↓                          ↓
  Forward pass →→→→→→→→→→→→→ Forward pass →→→→→→→→→→→→→ Forward pass
  (TensorShard via tunnel)   (TensorShard via tunnel)   (LayerOutput returned)
```

## Protocol

1. **Discovery**: PeerRegistry tracks each node's RAM, GPU TFLOPS, bandwidth
2. **Partitioning**: Coordinator splits model layers proportional to peer RAM
3. **Loading**: Each peer loads its assigned layer range into memory
4. **Inference**: Forward pass activations flow through compute tunnels as `TunnelPayload::TensorShard`
5. **Return**: Final node sends `TunnelPayload::LayerOutput` back to coordinator

## Prerequisites

The `TunnelPayload` enum already defines the needed message types:
- `TunnelPayload::TensorShard { request_id, layer_id, shard_data }`
- `TunnelPayload::LayerOutput { request_id, layer_range, activations }`

## Acceptance Criteria

- [ ] Model partitioner: split layers across N peers based on RAM
- [ ] Pipeline scheduler: overlap compute and network transfer
- [ ] TensorShard routing via compute tunnels (low-latency path)
- [ ] KV-cache sharing: cache intermediate results for repeated prompts
- [ ] Benchmark: 70B model inference across 3 nodes vs single-node OOM
- [ ] Graceful degradation: if peer disconnects mid-inference, retry on remaining peers

## Depends On

- #002 (P2P compute tunnels for tensor transport)
- #005 (Distributed AI inference framework)
- #014 (Inference revenue wiring)

## Progress

**Current**: tensor_parallel.rs (563 lines) — ModelPartitioner for layer-to-peer assignment based on VRAM, PipelineScheduler for overlapping compute/network transfer, KvCacheManager for intermediate result caching. TunnelPayload TensorShard/LayerOutput messages integrated.

## Files

- `crates/q-compute/src/tensor_parallel.rs` — ModelPartitioner, PipelineScheduler, KvCacheManager
- `crates/q-compute/src/tunnel.rs` — TensorShard/LayerOutput payload types
- `crates/q-ai-inference/src/distributed.rs` — Multi-node inference coordinator
