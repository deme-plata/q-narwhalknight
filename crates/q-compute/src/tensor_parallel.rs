//! # Issue #018: Cross-Node Tensor Parallelism for Large Model Inference
//!
//! Splits large AI model layers across multiple nodes connected via compute
//! tunnels, enabling inference of 70B+ parameter models that don't fit in
//! a single node's VRAM/RAM.
//!
//! ## Architecture
//!
//! ```text
//! User Request → Coordinator (partitions model)
//!   → Node A: Layers 0-23    (TensorShard via tunnel)
//!   → Node B: Layers 24-47   (TensorShard via tunnel)
//!   → Node C: Layers 48-69   (LayerOutput returned)
//! ```

#![allow(dead_code)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

/// A peer's available resources for tensor parallelism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerResources {
    pub peer_id: String,
    /// Available RAM in MB for model layers.
    pub ram_available_mb: u64,
    /// Available VRAM in MB (0 if no GPU).
    pub vram_available_mb: u64,
    /// Network bandwidth to this peer in Mbps.
    pub bandwidth_mbps: f64,
    /// Latency to this peer in ms.
    pub latency_ms: u32,
    /// Whether this peer is currently available.
    pub available: bool,
}

/// A layer range assigned to a peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerAssignment {
    pub peer_id: String,
    /// Inclusive start layer.
    pub start_layer: u32,
    /// Exclusive end layer.
    pub end_layer: u32,
    /// Estimated memory needed for these layers in MB.
    pub estimated_memory_mb: u64,
}

impl LayerAssignment {
    pub fn layer_count(&self) -> u32 {
        self.end_layer - self.start_layer
    }
}

/// A model partitioning plan across N peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionPlan {
    /// Model identifier.
    pub model_id: String,
    /// Total number of layers in the model.
    pub total_layers: u32,
    /// Total model size in MB.
    pub total_size_mb: u64,
    /// Per-peer layer assignments (ordered by layer range).
    pub assignments: Vec<LayerAssignment>,
    /// Estimated pipeline latency in ms (sum of inter-node transfers).
    pub estimated_pipeline_latency_ms: u64,
}

impl PartitionPlan {
    /// Check if all layers are covered exactly once.
    pub fn is_complete(&self) -> bool {
        if self.assignments.is_empty() {
            return false;
        }
        let mut covered = vec![false; self.total_layers as usize];
        for a in &self.assignments {
            for l in a.start_layer..a.end_layer {
                if (l as usize) < covered.len() {
                    if covered[l as usize] {
                        return false; // Overlapping.
                    }
                    covered[l as usize] = true;
                }
            }
        }
        covered.iter().all(|&c| c)
    }
}

/// Status of a distributed inference pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStatus {
    /// Waiting for all peers to load their layers.
    Loading,
    /// Forward pass in progress.
    Running,
    /// Completed successfully.
    Completed,
    /// A peer disconnected or errored.
    Failed,
}

/// A tensor shard being sent between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShard {
    /// Request ID for correlating forward pass stages.
    pub request_id: String,
    /// Which layer produced this shard.
    pub source_layer: u32,
    /// Destination layer (next in pipeline).
    pub dest_layer: u32,
    /// Serialized tensor data.
    pub data: Vec<u8>,
    /// Data format hint (e.g. "f32", "f16", "bf16").
    pub dtype: String,
    /// Tensor shape dimensions.
    pub shape: Vec<u64>,
}

/// A pipeline inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRequest {
    pub request_id: String,
    pub model_id: String,
    /// Input tokens (serialized).
    pub input_data: Vec<u8>,
    /// Current status.
    pub status: PipelineStatus,
    /// Which stage of the pipeline we're at (0-indexed peer assignment).
    pub current_stage: usize,
    /// Timestamp when request was created.
    pub created_at_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════
// Model Partitioner
// ═══════════════════════════════════════════════════════════════════

/// Partitions a model's layers across available peers based on their
/// RAM/VRAM capacity. Uses a greedy proportional allocation strategy.
pub struct ModelPartitioner;

impl ModelPartitioner {
    /// Partition a model across available peers.
    ///
    /// Strategy: allocate layers proportional to each peer's available memory.
    /// Peers with more RAM get more layers.
    pub fn partition(
        model_id: &str,
        total_layers: u32,
        total_size_mb: u64,
        peers: &[PeerResources],
    ) -> Result<PartitionPlan, String> {
        let available_peers: Vec<&PeerResources> = peers.iter()
            .filter(|p| p.available)
            .collect();

        if available_peers.is_empty() {
            return Err("No available peers for partitioning".to_string());
        }

        if total_layers == 0 {
            return Err("Model has 0 layers".to_string());
        }

        // Use the maximum of RAM and VRAM as the peer's capacity.
        let total_memory: u64 = available_peers.iter()
            .map(|p| p.ram_available_mb.max(p.vram_available_mb))
            .sum();

        if total_memory == 0 {
            return Err("No memory available across peers".to_string());
        }

        let size_per_layer = total_size_mb as f64 / total_layers as f64;
        let mut assignments = Vec::new();
        let mut current_layer = 0u32;

        for (i, peer) in available_peers.iter().enumerate() {
            let peer_memory = peer.ram_available_mb.max(peer.vram_available_mb);
            let proportion = peer_memory as f64 / total_memory as f64;

            let layers_for_peer = if i == available_peers.len() - 1 {
                // Last peer gets remaining layers.
                total_layers - current_layer
            } else {
                let raw = (total_layers as f64 * proportion).round() as u32;
                raw.max(1).min(total_layers - current_layer)
            };

            if layers_for_peer == 0 {
                continue;
            }

            let end_layer = current_layer + layers_for_peer;
            let estimated_memory = (layers_for_peer as f64 * size_per_layer) as u64;

            assignments.push(LayerAssignment {
                peer_id: peer.peer_id.clone(),
                start_layer: current_layer,
                end_layer,
                estimated_memory_mb: estimated_memory,
            });

            current_layer = end_layer;
        }

        // Estimate pipeline latency: sum of inter-node transfer times.
        let estimated_pipeline_latency_ms: u64 = available_peers.iter()
            .map(|p| p.latency_ms as u64)
            .sum::<u64>()
            .saturating_sub(available_peers.first().map(|p| p.latency_ms as u64).unwrap_or(0));

        let plan = PartitionPlan {
            model_id: model_id.to_string(),
            total_layers,
            total_size_mb,
            assignments,
            estimated_pipeline_latency_ms,
        };

        if !plan.is_complete() {
            return Err("Partition plan doesn't cover all layers".to_string());
        }

        info!(
            model = model_id,
            layers = total_layers,
            peers = available_peers.len(),
            latency_ms = plan.estimated_pipeline_latency_ms,
            "Model partitioned across {} peers",
            available_peers.len(),
        );

        Ok(plan)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline Scheduler
// ═══════════════════════════════════════════════════════════════════

/// Manages active distributed inference pipelines.
#[derive(Debug, Clone)]
pub struct PipelineScheduler {
    /// Active partition plans keyed by model_id.
    plans: Arc<RwLock<HashMap<String, PartitionPlan>>>,
    /// Active pipeline requests keyed by request_id.
    requests: Arc<RwLock<HashMap<String, PipelineRequest>>>,
    /// Total requests completed.
    total_completed: Arc<std::sync::atomic::AtomicU64>,
    /// Total requests failed.
    total_failed: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for PipelineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineScheduler {
    pub fn new() -> Self {
        Self {
            plans: Arc::new(RwLock::new(HashMap::new())),
            requests: Arc::new(RwLock::new(HashMap::new())),
            total_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Register a partition plan for a model.
    pub fn register_plan(&self, plan: PartitionPlan) {
        let model_id = plan.model_id.clone();
        self.plans.write().insert(model_id, plan);
    }

    /// Get the partition plan for a model.
    pub fn get_plan(&self, model_id: &str) -> Option<PartitionPlan> {
        self.plans.read().get(model_id).cloned()
    }

    /// Submit a new pipeline inference request.
    pub fn submit_request(&self, request_id: &str, model_id: &str, input_data: Vec<u8>) -> bool {
        if !self.plans.read().contains_key(model_id) {
            warn!(model_id, "No partition plan for model");
            return false;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let req = PipelineRequest {
            request_id: request_id.to_string(),
            model_id: model_id.to_string(),
            input_data,
            status: PipelineStatus::Loading,
            current_stage: 0,
            created_at_ms: now,
        };
        self.requests.write().insert(request_id.to_string(), req);
        true
    }

    /// Advance a pipeline request to the next stage.
    pub fn advance_stage(&self, request_id: &str) -> Option<PipelineStatus> {
        let mut requests = self.requests.write();
        if let Some(req) = requests.get_mut(request_id) {
            let plans = self.plans.read();
            if let Some(plan) = plans.get(&req.model_id) {
                if req.status == PipelineStatus::Loading {
                    req.status = PipelineStatus::Running;
                }
                req.current_stage += 1;
                if req.current_stage >= plan.assignments.len() {
                    req.status = PipelineStatus::Completed;
                    self.total_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Some(req.status);
            }
        }
        None
    }

    /// Mark a request as failed (peer disconnected, etc.).
    pub fn fail_request(&self, request_id: &str, _reason: &str) {
        if let Some(req) = self.requests.write().get_mut(request_id) {
            req.status = PipelineStatus::Failed;
            self.total_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get a pipeline request's status.
    pub fn get_request(&self, request_id: &str) -> Option<PipelineRequest> {
        self.requests.read().get(request_id).cloned()
    }

    /// Get the next peer that should process the current stage.
    pub fn next_peer_for_request(&self, request_id: &str) -> Option<String> {
        let requests = self.requests.read();
        let req = requests.get(request_id)?;
        let plans = self.plans.read();
        let plan = plans.get(&req.model_id)?;
        plan.assignments.get(req.current_stage).map(|a| a.peer_id.clone())
    }

    /// Statistics.
    pub fn active_requests(&self) -> usize {
        self.requests.read()
            .values()
            .filter(|r| matches!(r.status, PipelineStatus::Loading | PipelineStatus::Running))
            .count()
    }

    pub fn total_completed(&self) -> u64 {
        self.total_completed.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn total_failed(&self) -> u64 {
        self.total_failed.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peers(count: usize, ram_mb: u64) -> Vec<PeerResources> {
        (0..count).map(|i| PeerResources {
            peer_id: format!("peer-{}", i),
            ram_available_mb: ram_mb,
            vram_available_mb: 0,
            bandwidth_mbps: 1000.0,
            latency_ms: 10,
            available: true,
        }).collect()
    }

    #[test]
    fn test_partition_even_split() {
        let peers = make_peers(3, 1000);
        let plan = ModelPartitioner::partition("llama-70b", 69, 70_000, &peers).unwrap();
        assert_eq!(plan.assignments.len(), 3);
        assert!(plan.is_complete());
        // Each peer gets ~23 layers.
        let total: u32 = plan.assignments.iter().map(|a| a.layer_count()).sum();
        assert_eq!(total, 69);
    }

    #[test]
    fn test_partition_uneven_memory() {
        let peers = vec![
            PeerResources { peer_id: "big".into(), ram_available_mb: 3000, vram_available_mb: 0, bandwidth_mbps: 1000.0, latency_ms: 5, available: true },
            PeerResources { peer_id: "small".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 1000.0, latency_ms: 15, available: true },
        ];
        let plan = ModelPartitioner::partition("model-40", 40, 40_000, &peers).unwrap();
        assert!(plan.is_complete());
        // Big peer should get more layers.
        let big_layers = plan.assignments.iter().find(|a| a.peer_id == "big").unwrap().layer_count();
        let small_layers = plan.assignments.iter().find(|a| a.peer_id == "small").unwrap().layer_count();
        assert!(big_layers > small_layers, "big={big_layers} should be > small={small_layers}");
    }

    #[test]
    fn test_partition_single_peer() {
        let peers = make_peers(1, 8000);
        let plan = ModelPartitioner::partition("small-7b", 32, 7_000, &peers).unwrap();
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].start_layer, 0);
        assert_eq!(plan.assignments[0].end_layer, 32);
    }

    #[test]
    fn test_partition_no_peers() {
        let result = ModelPartitioner::partition("model", 32, 7000, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_filters_unavailable() {
        let peers = vec![
            PeerResources { peer_id: "a".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 5, available: true },
            PeerResources { peer_id: "b".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 5, available: false },
        ];
        let plan = ModelPartitioner::partition("model", 20, 10_000, &peers).unwrap();
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].peer_id, "a");
    }

    #[test]
    fn test_partition_plan_completeness() {
        let plan = PartitionPlan {
            model_id: "test".into(),
            total_layers: 10,
            total_size_mb: 1000,
            assignments: vec![
                LayerAssignment { peer_id: "a".into(), start_layer: 0, end_layer: 5, estimated_memory_mb: 500 },
                LayerAssignment { peer_id: "b".into(), start_layer: 5, end_layer: 10, estimated_memory_mb: 500 },
            ],
            estimated_pipeline_latency_ms: 20,
        };
        assert!(plan.is_complete());

        // Gap plan.
        let gap_plan = PartitionPlan {
            model_id: "test".into(),
            total_layers: 10,
            total_size_mb: 1000,
            assignments: vec![
                LayerAssignment { peer_id: "a".into(), start_layer: 0, end_layer: 3, estimated_memory_mb: 300 },
                LayerAssignment { peer_id: "b".into(), start_layer: 5, end_layer: 10, estimated_memory_mb: 500 },
            ],
            estimated_pipeline_latency_ms: 20,
        };
        assert!(!gap_plan.is_complete());
    }

    #[test]
    fn test_pipeline_full_lifecycle() {
        let scheduler = PipelineScheduler::new();

        // Register a plan.
        let peers = make_peers(3, 1000);
        let plan = ModelPartitioner::partition("llama-70b", 69, 70_000, &peers).unwrap();
        scheduler.register_plan(plan.clone());

        // Submit request.
        assert!(scheduler.submit_request("req-1", "llama-70b", vec![1, 2, 3]));
        assert_eq!(scheduler.active_requests(), 1);

        // Advance through all stages.
        for i in 0..plan.assignments.len() {
            let next_peer = scheduler.next_peer_for_request("req-1");
            assert!(next_peer.is_some(), "stage {i} should have next peer");
            let status = scheduler.advance_stage("req-1").unwrap();
            if i < plan.assignments.len() - 1 {
                assert_eq!(status, PipelineStatus::Running);
            } else {
                assert_eq!(status, PipelineStatus::Completed);
            }
        }

        assert_eq!(scheduler.total_completed(), 1);
        assert_eq!(scheduler.active_requests(), 0);
    }

    #[test]
    fn test_pipeline_fail_request() {
        let scheduler = PipelineScheduler::new();
        let peers = make_peers(2, 1000);
        let plan = ModelPartitioner::partition("model", 20, 10_000, &peers).unwrap();
        scheduler.register_plan(plan);

        scheduler.submit_request("req-1", "model", vec![]);
        scheduler.fail_request("req-1", "peer disconnected");

        let req = scheduler.get_request("req-1").unwrap();
        assert_eq!(req.status, PipelineStatus::Failed);
        assert_eq!(scheduler.total_failed(), 1);
    }

    #[test]
    fn test_pipeline_no_plan() {
        let scheduler = PipelineScheduler::new();
        assert!(!scheduler.submit_request("req-1", "no-such-model", vec![]));
    }

    #[test]
    fn test_tensor_shard_serde() {
        let shard = TensorShard {
            request_id: "req-1".into(),
            source_layer: 23,
            dest_layer: 24,
            data: vec![1, 2, 3, 4],
            dtype: "f16".into(),
            shape: vec![1, 512, 4096],
        };
        let json = serde_json::to_string(&shard).unwrap();
        let back: TensorShard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_layer, 23);
        assert_eq!(back.shape, vec![1, 512, 4096]);
    }

    #[test]
    fn test_partition_vram_preferred_over_ram() {
        let peers = vec![
            PeerResources { peer_id: "cpu".into(), ram_available_mb: 4000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 5, available: true },
            PeerResources { peer_id: "gpu".into(), ram_available_mb: 1000, vram_available_mb: 8000, bandwidth_mbps: 100.0, latency_ms: 5, available: true },
        ];
        let plan = ModelPartitioner::partition("model", 30, 15_000, &peers).unwrap();
        // GPU peer has 8000 VRAM (max of 1000 RAM, 8000 VRAM), CPU has 4000 RAM.
        // GPU should get ~2/3 of layers (8000/12000).
        let gpu_layers = plan.assignments.iter().find(|a| a.peer_id == "gpu").unwrap().layer_count();
        let cpu_layers = plan.assignments.iter().find(|a| a.peer_id == "cpu").unwrap().layer_count();
        assert!(gpu_layers > cpu_layers, "gpu={gpu_layers} should > cpu={cpu_layers}");
    }

    #[test]
    fn test_estimated_pipeline_latency() {
        let peers = vec![
            PeerResources { peer_id: "a".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 10, available: true },
            PeerResources { peer_id: "b".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 20, available: true },
            PeerResources { peer_id: "c".into(), ram_available_mb: 1000, vram_available_mb: 0, bandwidth_mbps: 100.0, latency_ms: 30, available: true },
        ];
        let plan = ModelPartitioner::partition("model", 30, 10_000, &peers).unwrap();
        // Pipeline latency = sum of all latencies minus the first (local node).
        // 10 + 20 + 30 - 10 = 50
        assert_eq!(plan.estimated_pipeline_latency_ms, 50);
    }
}
