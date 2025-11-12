# Q-NarwhalKnight Distributed AI Architecture
## Decentralized AI Inference with Mistral.rs + libp2p

**Goal**: Make AI common by decentralizing it - users contribute computing power for distributed AI model inference using Mistral-7B-Instruct-v0.3 across the Q-NarwhalKnight network.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Q-NarwhalKnight Network                          │
│                   (libp2p + Kademlia DHT)                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐      ┌───────▼────────┐      ┌──────▼─────────┐
│   Node A       │      │   Node B       │      │   Node C       │
│  (Layers 0-10) │◄────►│  (Layers 11-21)│◄────►│  (Layers 22-32)│
│  Mistral.rs    │      │  Mistral.rs    │      │  Mistral.rs    │
│  Worker        │      │  Worker        │      │  Worker        │
└────────────────┘      └────────────────┘      └────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  Coordinator    │
                        │  Node           │
                        │  (Orchestrator) │
                        └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  User Request   │
                        │  (API/Frontend) │
                        └─────────────────┘
```

---

## 🎯 Key Concepts

### Mistral.rs Analysis

From analyzing the codebase:

1. **Distributed Backends**:
   - **NCCL**: CUDA-only, high-performance GPU communication (not suitable for heterogeneous network)
   - **Ring Backend**: TCP-based, heterogeneous, works on any device (CPU, Metal, CUDA)
   - Uses ring topology for tensor parallelism across nodes

2. **Layer Sharding**:
   - Mistral-7B has 32 transformer layers
   - Each layer can be split across different nodes
   - Supports tensor parallelism where model weights are distributed
   - Uses `ShardedVarBuilder` to load model shards

3. **Communication Pattern**:
   - Master/worker architecture (rank 0 is master)
   - TCP ring topology for collective operations
   - IPC (Inter-Process Communication) for local coordination
   - Request/response pattern with async channels

4. **Request Flow**:
   - Tokenization on master node
   - Forward passes distributed across ring
   - Collective all-reduce operations for tensor synchronization
   - Detokenization on master node

### Q-NarwhalKnight Integration Points

Your network already has:
- ✅ **libp2p Unified Network Manager** (`q-network` crate)
- ✅ **Kademlia DHT** for peer discovery
- ✅ **Gossipsub** for pub/sub messaging (`/qnk/testnet/blocks`, `/qnk/testnet/transactions`)
- ✅ **Tor integration** for privacy
- ✅ **P2P connection management**

---

## 🏗️ Phase-Based Implementation Plan

---

## **Phase 0: Foundation & Analysis** ✅ COMPLETED

**Goal**: Understand the architecture and prepare infrastructure

### Tasks Completed:
- ✅ Cloned and analyzed mistral.rs repository
- ✅ Studied distributed backends (NCCL vs Ring)
- ✅ Identified integration points with Q-NarwhalKnight
- ✅ Reviewed existing libp2p infrastructure

### Key Findings:
- **Ring Backend** is the optimal choice (heterogeneous, TCP-based)
- Mistral-7B-Instruct-v0.3-GGUF is a good target model (quantized, efficient)
- Current libp2p Gossipsub can be extended for AI inference messages

---

## **Phase 1: Infrastructure Setup** (Weeks 1-2)

**Goal**: Create the foundational crate and integrate mistral.rs

### 1.1 Create `q-ai-inference` Crate

```bash
cd /opt/orobit/shared/q-narwhalknight
cargo new --lib crates/q-ai-inference
```

**Cargo.toml additions**:
```toml
[workspace]
members = [
    # ... existing crates ...
    "crates/q-ai-inference",
]

[dependencies]
mistralrs = "0.6.0"
mistralrs-core = "0.6.0"
candle-core = { git = "https://github.com/EricLBuehler/candle.git", version = "0.9.1" }
candle-nn = { git = "https://github.com/EricLBuehler/candle.git", version = "0.9.1" }
libp2p = { version = "0.54", features = ["gossipsub", "kad", "tcp", "noise", "mplex", "yamux"] }
tokio = { version = "1.45", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
anyhow = "1.0"
```

### 1.2 Design Core Data Structures

**File: `crates/q-ai-inference/src/types.rs`**

```rust
use serde::{Deserialize, Serialize};

/// AI inference request from user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub model: String, // "mistral-7b-instruct-v0.3"
}

/// Response from distributed inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub generated_text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
}

/// Layer assignment for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub node_id: String,
    pub peer_id: String,
    pub layer_start: usize, // e.g., 0
    pub layer_end: usize,   // e.g., 10 (layers 0-10 = 11 layers)
    pub device_capability: DeviceCapability,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceCapability {
    CPU { cores: usize, ram_gb: usize },
    CUDA { vram_gb: usize, compute_capability: String },
    Metal { vram_gb: usize },
}

/// Tensor data for layer communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub layer_index: usize,
    pub data: Vec<f32>, // Serialized tensor (consider compression)
    pub shape: Vec<usize>,
}

/// Messages for Gossipsub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIMessage {
    InferenceRequest(InferenceRequest),
    InferenceResponse(InferenceResponse),
    LayerOutput(TensorData),
    NodeCapability(LayerAssignment),
    CoordinatorElection { node_id: String, score: u64 },
}
```

### 1.3 Download and Store Model

```bash
# Create model storage directory
mkdir -p /opt/orobit/shared/q-narwhalknight/models

# Download Mistral-7B-Instruct-v0.3-GGUF (Q4_K_M quantization ~4GB)
cd /opt/orobit/shared/q-narwhalknight/models
wget https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

# Verify download
ls -lh Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
```

**Deliverables**:
- ✅ `q-ai-inference` crate created
- ✅ Core data structures defined
- ✅ Mistral-7B model downloaded

---

## **Phase 2: Gossipsub AI Topics** (Weeks 3-4)

**Goal**: Extend Gossipsub for AI inference communication

### 2.1 Add New Gossipsub Topics

**File: `crates/q-network/src/unified_network_manager.rs`** (modify existing)

Add AI-specific topics:
```rust
pub const AI_INFERENCE_REQUEST_TOPIC: &str = "/qnk/testnet/ai-inference-request";
pub const AI_LAYER_OUTPUT_TOPIC: &str = "/qnk/testnet/ai-layer-output";
pub const AI_NODE_CAPABILITY_TOPIC: &str = "/qnk/testnet/ai-node-capability";
pub const AI_COORDINATOR_TOPIC: &str = "/qnk/testnet/ai-coordinator";
```

Subscribe to AI topics in initialization:
```rust
impl UnifiedNetworkManager {
    pub async fn new(config: NetworkConfig) -> Result<Self> {
        // ... existing initialization ...

        // Subscribe to AI inference topics
        gossipsub.subscribe(&IdentTopic::new(AI_INFERENCE_REQUEST_TOPIC))?;
        gossipsub.subscribe(&IdentTopic::new(AI_LAYER_OUTPUT_TOPIC))?;
        gossipsub.subscribe(&IdentTopic::new(AI_NODE_CAPABILITY_TOPIC))?;
        gossipsub.subscribe(&IdentTopic::new(AI_COORDINATOR_TOPIC))?;

        info!("📢 Subscribed to AI inference Gossipsub topics");

        // ... rest of initialization ...
    }
}
```

### 2.2 AI Message Handler

**File: `crates/q-ai-inference/src/gossipsub_handler.rs`**

```rust
use crate::types::AIMessage;
use libp2p::gossipsub::Event as GossipsubEvent;
use tokio::sync::mpsc;
use tracing::{error, info};

pub struct AIGossipsubHandler {
    ai_message_tx: mpsc::Sender<AIMessage>,
}

impl AIGossipsubHandler {
    pub fn new(ai_message_tx: mpsc::Sender<AIMessage>) -> Self {
        Self { ai_message_tx }
    }

    pub async fn handle_gossipsub_event(&self, event: GossipsubEvent) {
        match event {
            GossipsubEvent::Message {
                propagation_source,
                message_id,
                message,
            } => {
                let topic = message.topic.as_str();

                if topic.contains("ai-inference") || topic.contains("ai-layer")
                    || topic.contains("ai-node-capability") || topic.contains("ai-coordinator") {

                    match serde_json::from_slice::<AIMessage>(&message.data) {
                        Ok(ai_msg) => {
                            info!("🤖 Received AI message from {}: {:?}", propagation_source, ai_msg);

                            if let Err(e) = self.ai_message_tx.send(ai_msg).await {
                                error!("Failed to forward AI message: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize AI message: {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
```

**Deliverables**:
- ✅ New Gossipsub topics for AI inference
- ✅ AI message handler integrated into libp2p event loop

---

## **Phase 3: Node Capability Discovery** (Weeks 5-6)

**Goal**: Nodes advertise their computational capabilities via Kademlia DHT

### 3.1 Device Capability Detection

**File: `crates/q-ai-inference/src/capability_detector.rs`**

```rust
use crate::types::{DeviceCapability, LayerAssignment};
use sysinfo::{System, SystemExt, CpuExt};
use anyhow::Result;

pub struct CapabilityDetector;

impl CapabilityDetector {
    pub fn detect() -> Result<DeviceCapability> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_cores = sys.cpus().len();
        let ram_gb = (sys.total_memory() / 1024 / 1024 / 1024) as usize;

        // Check for CUDA (requires cuda-sys or similar)
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_info) = detect_cuda() {
                return Ok(DeviceCapability::CUDA {
                    vram_gb: cuda_info.vram_gb,
                    compute_capability: cuda_info.compute_capability,
                });
            }
        }

        // Check for Metal (macOS)
        #[cfg(target_os = "macos")]
        {
            if let Some(metal_info) = detect_metal() {
                return Ok(DeviceCapability::Metal {
                    vram_gb: metal_info.vram_gb,
                });
            }
        }

        // Default to CPU
        Ok(DeviceCapability::CPU { cores: cpu_cores, ram_gb })
    }

    pub fn estimate_layer_capacity(capability: &DeviceCapability) -> usize {
        match capability {
            DeviceCapability::CPU { cores, ram_gb } => {
                // Conservative estimate: ~1 layer per 4GB RAM, max 8 layers
                (*ram_gb / 4).min(8).max(1)
            }
            DeviceCapability::CUDA { vram_gb, .. } => {
                // ~1 layer per 1GB VRAM for Q4 quantization
                (*vram_gb).min(32).max(2)
            }
            DeviceCapability::Metal { vram_gb } => {
                // Similar to CUDA
                (*vram_gb).min(32).max(2)
            }
        }
    }
}
```

### 3.2 Capability Announcement

**File: `crates/q-ai-inference/src/node_announcer.rs`**

```rust
use crate::types::{AIMessage, LayerAssignment};
use crate::capability_detector::CapabilityDetector;
use libp2p::gossipsub::{Behaviour as Gossipsub, IdentTopic};
use std::time::Duration;
use tokio::time;
use tracing::info;

pub struct NodeAnnouncer {
    node_id: String,
    peer_id: String,
    capability: DeviceCapability,
}

impl NodeAnnouncer {
    pub fn new(node_id: String, peer_id: String) -> Self {
        let capability = CapabilityDetector::detect().unwrap_or_else(|_| {
            DeviceCapability::CPU { cores: 2, ram_gb: 4 }
        });

        Self { node_id, peer_id, capability }
    }

    pub async fn start_announcing(&self, gossipsub: &mut Gossipsub) {
        let mut interval = time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            let capacity = CapabilityDetector::estimate_layer_capacity(&self.capability);

            let assignment = LayerAssignment {
                node_id: self.node_id.clone(),
                peer_id: self.peer_id.clone(),
                layer_start: 0, // Will be assigned by coordinator
                layer_end: 0,
                device_capability: self.capability.clone(),
            };

            let message = AIMessage::NodeCapability(assignment);
            let topic = IdentTopic::new(AI_NODE_CAPABILITY_TOPIC);

            if let Ok(data) = serde_json::to_vec(&message) {
                if let Err(e) = gossipsub.publish(topic, data) {
                    error!("Failed to announce node capability: {}", e);
                } else {
                    info!("🤖 Announced node capability: {} can handle {} layers",
                          self.node_id, capacity);
                }
            }
        }
    }
}
```

**Deliverables**:
- ✅ Device capability detection (CPU, CUDA, Metal)
- ✅ Periodic capability announcements via Gossipsub
- ✅ DHT storage of node capabilities

---

## **Phase 4: Coordinator Node** (Weeks 7-9)

**Goal**: Implement coordinator election and layer assignment orchestration

### 4.1 Coordinator Election

**File: `crates/q-ai-inference/src/coordinator_election.rs`**

```rust
use crate::types::AIMessage;
use libp2p::PeerId;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

pub struct CoordinatorElection {
    current_coordinator: RwLock<Option<String>>,
    node_scores: RwLock<HashMap<String, (u64, Instant)>>,
    my_node_id: String,
}

impl CoordinatorElection {
    pub fn new(node_id: String) -> Self {
        Self {
            current_coordinator: RwLock::new(None),
            node_scores: RwLock::new(HashMap::new()),
            my_node_id: node_id,
        }
    }

    /// Calculate election score based on:
    /// - Uptime
    /// - Number of successful inferences
    /// - Network connectivity (peer count)
    pub fn calculate_score(&self, uptime_secs: u64, inference_count: u64, peer_count: usize) -> u64 {
        uptime_secs * 10 + inference_count * 100 + (peer_count as u64) * 50
    }

    pub async fn process_election_message(&self, node_id: String, score: u64) {
        let mut scores = self.node_scores.write().await;
        scores.insert(node_id, (score, Instant::now()));

        // Remove stale scores (older than 60 seconds)
        scores.retain(|_, (_, timestamp)| timestamp.elapsed() < Duration::from_secs(60));

        // Elect coordinator (highest score)
        if let Some((coordinator, _)) = scores.iter()
            .max_by_key(|(_, (score, _))| score) {

            let mut current = self.current_coordinator.write().await;
            if current.as_ref() != Some(coordinator) {
                info!("🎯 New coordinator elected: {}", coordinator);
                *current = Some(coordinator.clone());
            }
        }
    }

    pub async fn am_i_coordinator(&self) -> bool {
        let coordinator = self.current_coordinator.read().await;
        coordinator.as_ref() == Some(&self.my_node_id)
    }
}
```

### 4.2 Layer Assignment Orchestrator

**File: `crates/q-ai-inference/src/layer_orchestrator.rs`**

```rust
use crate::types::{LayerAssignment, DeviceCapability};
use crate::capability_detector::CapabilityDetector;
use std::collections::HashMap;
use tracing::{info, warn};

pub struct LayerOrchestrator {
    total_layers: usize, // Mistral-7B has 32 layers
    available_nodes: HashMap<String, LayerAssignment>,
}

impl LayerOrchestrator {
    pub fn new(total_layers: usize) -> Self {
        Self {
            total_layers,
            available_nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, assignment: LayerAssignment) {
        self.available_nodes.insert(assignment.node_id.clone(), assignment);
    }

    /// Assign layers to nodes based on their capabilities
    /// Strategy: Greedy assignment prioritizing GPU nodes
    pub fn assign_layers(&mut self) -> Vec<LayerAssignment> {
        let mut assignments = Vec::new();

        // Sort nodes by capability (GPU > CPU, more RAM > less RAM)
        let mut nodes: Vec<_> = self.available_nodes.values().cloned().collect();
        nodes.sort_by(|a, b| {
            let a_score = self.capability_score(&a.device_capability);
            let b_score = self.capability_score(&b.device_capability);
            b_score.cmp(&a_score)
        });

        let mut current_layer = 0;

        for node in nodes.iter_mut() {
            if current_layer >= self.total_layers {
                break;
            }

            let capacity = CapabilityDetector::estimate_layer_capacity(&node.device_capability);
            let layers_to_assign = capacity.min(self.total_layers - current_layer);

            node.layer_start = current_layer;
            node.layer_end = current_layer + layers_to_assign - 1;

            info!("🎯 Assigned layers {}-{} to node {}",
                  node.layer_start, node.layer_end, node.node_id);

            assignments.push(node.clone());
            current_layer += layers_to_assign;
        }

        if current_layer < self.total_layers {
            warn!("⚠️ Not enough nodes to cover all {} layers! Only covered {}",
                  self.total_layers, current_layer);
        }

        assignments
    }

    fn capability_score(&self, capability: &DeviceCapability) -> u64 {
        match capability {
            DeviceCapability::CPU { cores, ram_gb } => {
                (*cores as u64) * 10 + (*ram_gb as u64)
            }
            DeviceCapability::CUDA { vram_gb, .. } => {
                (*vram_gb as u64) * 1000 // Heavily prioritize CUDA
            }
            DeviceCapability::Metal { vram_gb } => {
                (*vram_gb as u64) * 800 // Prioritize Metal but less than CUDA
            }
        }
    }
}
```

**Deliverables**:
- ✅ Coordinator election based on node scores
- ✅ Layer assignment orchestrator
- ✅ Dynamic rebalancing when nodes join/leave

---

## **Phase 5: Mistral.rs Worker Integration** (Weeks 10-12)

**Goal**: Each node loads assigned model layers and processes tensor data

### 5.1 Model Loader with Layer Selection

**File: `crates/q-ai-inference/src/model_loader.rs`**

```rust
use candle_core::{Device, Tensor};
use mistralrs_core::{ModelPaths, DeviceMapSetting};
use anyhow::Result;
use tracing::info;

pub struct QNKModelLoader {
    model_path: String,
    layer_start: usize,
    layer_end: usize,
    device: Device,
}

impl QNKModelLoader {
    pub fn new(model_path: String, layer_start: usize, layer_end: usize) -> Result<Self> {
        let device = Device::Cpu; // Or Device::Cuda(0) / Device::Metal(0)

        Ok(Self {
            model_path,
            layer_start,
            layer_end,
            device,
        })
    }

    pub async fn load_model_layers(&self) -> Result<MistralModel> {
        info!("🔧 Loading Mistral-7B layers {}-{}", self.layer_start, self.layer_end);

        // Use mistralrs-core to load only assigned layers
        // This requires custom layer loading logic

        // For now, load full model (Phase 6 will optimize this)
        let model_paths = ModelPaths::from_gguf_file(self.model_path.clone())?;

        // Load model with device mapping
        let device_map = DeviceMapSetting::Layers(vec![
            format!("{}:{}", 0, self.layer_end - self.layer_start)
        ]);

        // ... mistralrs loading logic ...

        info!("✅ Model layers loaded successfully");

        Ok(MistralModel { /* ... */ })
    }

    pub async fn forward_pass(&self, input_tensor: Tensor) -> Result<Tensor> {
        // Run forward pass through assigned layers
        // This will be the core inference logic

        todo!("Implement forward pass")
    }
}
```

### 5.2 Inference Worker

**File: `crates/q-ai-inference/src/inference_worker.rs`**

```rust
use crate::types::{AIMessage, InferenceRequest, TensorData};
use crate::model_loader::QNKModelLoader;
use libp2p::gossipsub::{Behaviour as Gossipsub, IdentTopic};
use tokio::sync::mpsc;
use tracing::{error, info};

pub struct InferenceWorker {
    model_loader: QNKModelLoader,
    layer_start: usize,
    layer_end: usize,
}

impl InferenceWorker {
    pub fn new(model_loader: QNKModelLoader, layer_start: usize, layer_end: usize) -> Self {
        Self {
            model_loader,
            layer_start,
            layer_end,
        }
    }

    pub async fn process_inference_request(
        &self,
        request: InferenceRequest,
        input_tensor: TensorData,
        gossipsub: &mut Gossipsub,
    ) -> Result<(), anyhow::Error> {
        info!("🤖 Processing inference for request {}: layers {}-{}",
              request.request_id, self.layer_start, self.layer_end);

        // Convert TensorData to candle Tensor
        let input = self.tensor_data_to_candle(&input_tensor)?;

        // Run forward pass through assigned layers
        let output = self.model_loader.forward_pass(input).await?;

        // Convert output back to TensorData
        let output_data = self.candle_to_tensor_data(&output)?;

        // Publish to next layer or final response
        let next_layer = self.layer_end + 1;

        if next_layer < 32 { // Mistral-7B has 32 layers
            // Send to next layer via Gossipsub
            let message = AIMessage::LayerOutput(output_data);
            let topic = IdentTopic::new(AI_LAYER_OUTPUT_TOPIC);

            let data = serde_json::to_vec(&message)?;
            gossipsub.publish(topic, data)?;

            info!("✅ Forwarded output to layer {}", next_layer);
        } else {
            // This was the final layer - send response
            info!("✅ Final layer complete - generating response");

            // Detokenize and send response
            // (This would be done by the coordinator node)
        }

        Ok(())
    }

    fn tensor_data_to_candle(&self, data: &TensorData) -> Result<Tensor> {
        // Convert serialized tensor data to candle Tensor
        todo!("Implement tensor conversion")
    }

    fn candle_to_tensor_data(&self, tensor: &Tensor) -> Result<TensorData> {
        // Convert candle Tensor to serialized TensorData
        todo!("Implement tensor conversion")
    }
}
```

**Deliverables**:
- ✅ Model loader for assigned layer ranges
- ✅ Inference worker processing layer computations
- ✅ Tensor serialization/deserialization for Gossipsub

---

## **Phase 6: End-to-End Pipeline** (Weeks 13-15)

**Goal**: Complete inference pipeline from user request to response

### 6.1 Coordinator Inference Manager

**File: `crates/q-ai-inference/src/coordinator_manager.rs`**

```rust
use crate::types::{InferenceRequest, InferenceResponse, AIMessage};
use crate::layer_orchestrator::LayerOrchestrator;
use libp2p::gossipsub::{Behaviour as Gossipsub, IdentTopic};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::info;

pub struct CoordinatorManager {
    orchestrator: RwLock<LayerOrchestrator>,
    active_requests: RwLock<HashMap<String, InferenceRequest>>,
}

impl CoordinatorManager {
    pub fn new() -> Self {
        Self {
            orchestrator: RwLock::new(LayerOrchestrator::new(32)), // Mistral-7B: 32 layers
            active_requests: RwLock::new(HashMap::new()),
        }
    }

    pub async fn handle_user_request(
        &self,
        request: InferenceRequest,
        gossipsub: &mut Gossipsub,
    ) -> Result<(), anyhow::Error> {
        info!("🎯 Coordinator received inference request: {}", request.request_id);

        // Store active request
        self.active_requests.write().await.insert(request.request_id.clone(), request.clone());

        // Get layer assignments
        let assignments = self.orchestrator.read().await.assign_layers();

        if assignments.is_empty() {
            error!("❌ No nodes available for inference!");
            return Err(anyhow::anyhow!("No nodes available"));
        }

        // Tokenize prompt (using mistralrs tokenizer)
        let tokens = self.tokenize_prompt(&request.prompt).await?;

        // Create initial tensor data
        let initial_tensor = TensorData {
            layer_index: 0,
            data: tokens,
            shape: vec![1, tokens.len()],
        };

        // Publish to first layer workers via Gossipsub
        let message = AIMessage::InferenceRequest(request);
        let topic = IdentTopic::new(AI_INFERENCE_REQUEST_TOPIC);

        let data = serde_json::to_vec(&message)?;
        gossipsub.publish(topic, data)?;

        info!("✅ Dispatched inference request to network");

        Ok(())
    }

    pub async fn handle_final_output(&self, output: TensorData) -> Result<InferenceResponse> {
        // Detokenize final layer output
        let text = self.detokenize_output(&output.data).await?;

        // Create response
        let response = InferenceResponse {
            request_id: "...".to_string(),
            generated_text: text,
            tokens_generated: output.data.len(),
            latency_ms: 0, // Calculate actual latency
        };

        Ok(response)
    }

    async fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        // Use mistralrs tokenizer
        todo!("Implement tokenization")
    }

    async fn detokenize_output(&self, tokens: &[f32]) -> Result<String> {
        // Use mistralrs detokenizer
        todo!("Implement detokenization")
    }
}
```

### 6.2 API Endpoint Integration

**File: `crates/q-api-server/src/handlers_ai.rs`** (new file)

```rust
use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use crate::QApiServerState;

#[derive(Debug, Deserialize)]
pub struct AIInferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct AIInferenceResponse {
    pub request_id: String,
    pub generated_text: String,
    pub latency_ms: u64,
}

pub async fn handle_ai_inference(
    State(state): State<QApiServerState>,
    Json(request): Json<AIInferenceRequest>,
) -> Result<Json<AIInferenceResponse>, StatusCode> {
    // Forward to distributed AI inference coordinator
    let inference_req = q_ai_inference::types::InferenceRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        prompt: request.prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        model: "mistral-7b-instruct-v0.3".to_string(),
    };

    // Send via coordinator manager
    // ... coordinator logic ...

    Ok(Json(AIInferenceResponse {
        request_id: inference_req.request_id,
        generated_text: "Generated response...".to_string(),
        latency_ms: 1000,
    }))
}
```

**Add to `main.rs` routes**:
```rust
.route("/api/v1/ai/inference", post(handlers_ai::handle_ai_inference))
```

**Deliverables**:
- ✅ Complete inference pipeline
- ✅ Tokenization/detokenization on coordinator
- ✅ REST API endpoint for user requests

---

## **Phase 7: Optimization & Production** (Weeks 16-20)

**Goal**: Optimize performance, add monitoring, handle failures

### 7.1 Tensor Compression

**File: `crates/q-ai-inference/src/tensor_compression.rs`**

```rust
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;

pub struct TensorCompression;

impl TensorCompression {
    pub fn compress(data: &[f32]) -> Result<Vec<u8>> {
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&bytes)?;

        Ok(encoder.finish()?)
    }

    pub fn decompress(compressed: &[u8]) -> Result<Vec<f32>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;

        let floats: Vec<f32> = bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }
}
```

### 7.2 Fault Tolerance & Retries

```rust
pub struct FaultTolerantInference {
    max_retries: usize,
    timeout: Duration,
}

impl FaultTolerantInference {
    pub async fn inference_with_retry(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse> {
        let mut attempts = 0;

        loop {
            match self.try_inference(&request).await {
                Ok(response) => return Ok(response),
                Err(e) if attempts < self.max_retries => {
                    warn!("Inference attempt {} failed: {}, retrying...", attempts + 1, e);
                    attempts += 1;
                    tokio::time::sleep(Duration::from_secs(2_u64.pow(attempts as u32))).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```

### 7.3 Monitoring Dashboard

**Prometheus Metrics**:
```rust
use prometheus::{IntCounter, Histogram, Registry};

pub struct AIMetrics {
    pub inference_requests: IntCounter,
    pub inference_latency: Histogram,
    pub node_participation: IntGaugeVec,
    pub layer_processing_time: HistogramVec,
}

impl AIMetrics {
    pub fn new(registry: &Registry) -> Self {
        // Register Prometheus metrics
        Self {
            inference_requests: IntCounter::new("ai_inference_requests_total", "Total inference requests")?,
            inference_latency: Histogram::new("ai_inference_latency_seconds", "Inference latency")?,
            // ... other metrics ...
        }
    }
}
```

**Deliverables**:
- ✅ Tensor compression for reduced bandwidth
- ✅ Retry logic and timeout handling
- ✅ Prometheus metrics for monitoring
- ✅ Grafana dashboard for visualization

---

## **Phase 8: Frontend Integration** (Weeks 21-22)

**Goal**: User-facing AI chat interface

### 8.1 React Component

**File: `gui/quantum-wallet/src/components/AIChat.tsx`**

```typescript
import React, { useState } from 'react';
import { Send } from 'lucide-react';

export const AIChat: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch('/api/v1/ai/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens: 512,
          temperature: 0.7,
        }),
      });

      const data = await res.json();
      setResponse(data.generated_text);
    } catch (error) {
      console.error('AI inference error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ai-chat-container">
      <h2>🤖 Distributed AI Inference</h2>
      <p className="text-sm text-gray-400">
        Powered by Mistral-7B across the Q-NarwhalKnight network
      </p>

      <form onSubmit={handleSubmit} className="mt-4">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          className="w-full p-3 bg-gray-800 border border-quantum-cyan rounded-lg"
          rows={4}
        />

        <button
          type="submit"
          disabled={loading}
          className="mt-2 px-6 py-3 bg-quantum-cyan text-white rounded-lg hover:bg-quantum-purple"
        >
          {loading ? 'Generating...' : 'Send'}
          <Send className="inline ml-2" />
        </button>
      </form>

      {response && (
        <div className="mt-6 p-4 bg-gray-800 rounded-lg">
          <h3 className="text-quantum-green mb-2">Response:</h3>
          <p className="whitespace-pre-wrap">{response}</p>
        </div>
      )}
    </div>
  );
};
```

**Deliverables**:
- ✅ AI chat interface in quantum wallet
- ✅ Real-time inference with loading states
- ✅ Response display and history

---

## 🎯 Success Metrics

### Technical Metrics:
- **Inference Latency**: < 5 seconds for 512 tokens
- **Network Utilization**: > 70% of available nodes participating
- **Fault Tolerance**: < 5% failed requests
- **Throughput**: > 10 concurrent inferences

### User Metrics:
- **Node Participation**: > 100 nodes contributing compute
- **Daily Inferences**: > 1000 requests/day
- **User Satisfaction**: > 4.5/5 rating

---

## 🔐 Security Considerations

### 1. Privacy
- **Tor Integration**: All AI inference requests routed through Tor
- **ZK-STARK Proofs**: User prompts encrypted with zero-knowledge proofs
- **No Data Persistence**: Inference data not stored permanently

### 2. Anti-Abuse
- **Rate Limiting**: Max 10 requests/minute per wallet
- **Proof-of-Work**: Small PoW required for inference requests
- **Reputation System**: Nodes ranked by reliability

### 3. Model Integrity
- **Checksum Verification**: Model weights verified before loading
- **Signed Layers**: Each layer signed by trusted authority
- **Byzantine Fault Tolerance**: Detect and exclude malicious nodes

---

## 📊 Resource Requirements

### Per Node:
- **RAM**: 4-8 GB (for Q4_K_M quantization)
- **Storage**: 5 GB (for model + caches)
- **Bandwidth**: 10 Mbps (for tensor data transfer)
- **CPU/GPU**: 4 cores or 1 GPU with 4GB VRAM

### Network-Wide:
- **Minimum Nodes**: 8 nodes (4 layers each)
- **Recommended Nodes**: 16+ nodes for redundancy
- **Coordinator Nodes**: 3-5 for election redundancy

---

## 🚀 Deployment Strategy

### Alpha Testing (Phase 1-4):
- Deploy on 3-5 internal test nodes
- Verify layer assignments and Gossipsub communication
- Test coordinator election

### Beta Testing (Phase 5-7):
- Open to 20-50 community nodes
- Collect performance metrics
- Iterate on optimization

### Production Launch (Phase 8):
- Full network deployment
- Frontend integration
- Public announcement

---

## 📚 Documentation Needed

1. **User Guide**: How to contribute compute power
2. **Developer Guide**: API reference and integration
3. **Architecture Diagrams**: Visual system overview
4. **Troubleshooting**: Common issues and solutions

---

## 🎓 Future Enhancements

### Phase 9+ (Future):
- **Multiple Models**: Support GPT-2, Llama-3, etc.
- **Model Fine-tuning**: Distributed training on user data
- **Incentives**: QNK token rewards for compute contribution
- **Speculative Decoding**: Speed up inference with speculation
- **Multi-modal**: Add vision and audio models

---

## 📝 Notes

### Why Ring Backend?
- **Heterogeneous**: Works on any device (CPU, CUDA, Metal)
- **TCP-based**: Easy integration with libp2p
- **No NCCL dependency**: More portable

### Why Mistral-7B-Instruct-v0.3?
- **Size**: 7B parameters (manageable for distributed inference)
- **Quality**: High-quality instruction following
- **GGUF**: Efficient quantization format
- **License**: Apache 2.0 (permissive)

### Why Layer Parallelism?
- **Simplicity**: Easier than tensor parallelism
- **Fault Tolerance**: Node failures only affect their layers
- **Scalability**: Add more nodes = more layers covered

---

## 🤝 Community Contribution

Users earn QNK tokens for:
- **Compute Contribution**: Proportional to layers processed
- **Uptime**: Bonus for high availability
- **Response Quality**: Validated by consensus

This creates a **decentralized AI commons** where users democratize access to AI inference!

---

**Status**: Phase 0 Complete ✅
**Next Steps**: Begin Phase 1 - Infrastructure Setup
**Estimated Total Timeline**: 22 weeks (~5.5 months)
**Risk Level**: Medium (requires coordination of distributed systems)

**Let's make AI common! 🚀🤖**
