//! Layer Assignment Algorithm for Distributed AI Inference
//!
//! This module implements the algorithm for distributing model layers across
//! participating nodes in the network. The assignment considers:
//! - Node hardware capabilities (VRAM, RAM, compute power)
//! - Network topology and latency
//! - Load balancing and fairness
//! - Fault tolerance and redundancy
//!
//! For Mistral-7B-Instruct-v0.3 (32 transformer layers):
//! - Input embedding layer (layer 0)
//! - 32 transformer blocks (layers 1-32)
//! - Output/LM head layer (layer 33)
//! Total: 34 computational units to distribute

use crate::coordinator_election::ElectionCandidate;
use crate::types::{DeviceCapability, LayerAssignment};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Layer assignment plan for distributed inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignmentPlan {
    /// Model being executed
    pub model_name: String,

    /// Total number of layers in the model
    pub total_layers: usize,

    /// Layer assignments (node_id -> LayerAssignment)
    pub assignments: HashMap<String, LayerAssignment>,

    /// Estimated total inference time (ms)
    pub estimated_latency_ms: u64,

    /// Plan creation timestamp
    pub created_at: i64,
}

impl LayerAssignmentPlan {
    /// Create a new empty assignment plan
    pub fn new(model_name: String, total_layers: usize) -> Self {
        Self {
            model_name,
            total_layers,
            assignments: HashMap::new(),
            estimated_latency_ms: 0,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Add a layer assignment to the plan
    pub fn add_assignment(&mut self, assignment: LayerAssignment) {
        self.assignments
            .insert(assignment.node_id.clone(), assignment);
    }

    /// Validate that all layers are assigned
    pub fn validate(&self) -> Result<()> {
        let mut assigned_layers = vec![false; self.total_layers];

        for assignment in self.assignments.values() {
            for layer in assignment.layer_start..=assignment.layer_end {
                if layer >= self.total_layers {
                    return Err(anyhow!(
                        "Layer index {} exceeds total layers {}",
                        layer,
                        self.total_layers
                    ));
                }
                if assigned_layers[layer] {
                    return Err(anyhow!("Layer {} assigned multiple times", layer));
                }
                assigned_layers[layer] = true;
            }
        }

        // Check all layers are assigned
        for (idx, assigned) in assigned_layers.iter().enumerate() {
            if !*assigned {
                return Err(anyhow!("Layer {} not assigned", idx));
            }
        }

        info!("✅ Layer assignment plan validated successfully");
        Ok(())
    }

    /// Get assignment for a specific node
    pub fn get_assignment(&self, node_id: &str) -> Option<&LayerAssignment> {
        self.assignments.get(node_id)
    }

    /// Get number of participating nodes
    pub fn node_count(&self) -> usize {
        self.assignments.len()
    }
}

/// Layer assignment coordinator
pub struct LayerAssignmentCoordinator {
    /// Model configuration
    model_name: String,
    total_layers: usize,
}

impl LayerAssignmentCoordinator {
    /// Create a new layer assignment coordinator
    pub fn new(model_name: String, total_layers: usize) -> Self {
        info!(
            "🎯 Initializing layer assignment coordinator for model: {} ({} layers)",
            model_name, total_layers
        );

        Self {
            model_name,
            total_layers,
        }
    }

    /// Create a Mistral-7B specific coordinator
    pub fn for_mistral_7b() -> Self {
        Self::new("mistral-7b-instruct-v0.3".to_string(), 34)
    }

    /// Assign layers to nodes based on their capabilities
    pub fn assign_layers(&self, candidates: Vec<ElectionCandidate>) -> Result<LayerAssignmentPlan> {
        if candidates.is_empty() {
            return Err(anyhow!("No candidates available for layer assignment"));
        }

        info!("📊 Assigning {} layers to {} nodes", self.total_layers, candidates.len());

        let mut plan = LayerAssignmentPlan::new(self.model_name.clone(), self.total_layers);

        // Sort candidates by capability score (descending)
        let mut sorted_candidates = candidates.clone();
        sorted_candidates.sort_by(|a, b| b.election_score.cmp(&a.election_score));

        // Calculate layer capacity for each node
        let mut node_capacities: Vec<(String, usize)> = sorted_candidates
            .iter()
            .map(|c| {
                let capacity = c.capability.estimate_layer_capacity();
                debug!(
                    "📦 Node {} capacity: {} layers (score: {})",
                    c.node_id, capacity, c.election_score
                );
                (c.node_id.clone(), capacity)
            })
            .collect();

        // Calculate total capacity
        let total_capacity: usize = node_capacities.iter().map(|(_, cap)| cap).sum();
        info!("📊 Total network capacity: {} layers", total_capacity);

        if total_capacity < self.total_layers {
            warn!(
                "⚠️ Total capacity {} is less than required layers {}",
                total_capacity, self.total_layers
            );
            // Scale up capacities proportionally
            let scale_factor = self.total_layers as f64 / total_capacity as f64;
            node_capacities = node_capacities
                .iter()
                .map(|(id, cap)| {
                    let scaled = ((*cap as f64) * scale_factor).ceil() as usize;
                    (id.clone(), scaled.max(1))
                })
                .collect();
        }

        // Assign layers sequentially based on capacity
        let mut current_layer = 0;
        for (_idx, (node_id, capacity)) in node_capacities.iter().enumerate() {
            if current_layer >= self.total_layers {
                break;
            }

            let candidate = sorted_candidates
                .iter()
                .find(|c| &c.node_id == node_id)
                .ok_or_else(|| anyhow!("Candidate not found: {}", node_id))?;

            let layers_to_assign = (*capacity).min(self.total_layers - current_layer);
            let layer_start = current_layer;
            let layer_end = current_layer + layers_to_assign - 1;

            let assignment = LayerAssignment {
                node_id: node_id.clone(),
                peer_id: candidate.peer_id.clone(),
                layer_start,
                layer_end,
                device_capability: candidate.capability.clone(),
                last_seen: chrono::Utc::now().timestamp(),
            };

            info!(
                "✅ Assigned layers {}-{} to node {} ({} layers)",
                layer_start, layer_end, node_id, layers_to_assign
            );

            plan.add_assignment(assignment);
            current_layer += layers_to_assign;
        }

        // Validate assignment plan
        plan.validate()?;

        // Estimate latency based on sequential processing
        plan.estimated_latency_ms = self.estimate_inference_latency(&plan, &sorted_candidates);

        info!(
            "🎉 Layer assignment complete: {} nodes, estimated latency: {}ms",
            plan.node_count(),
            plan.estimated_latency_ms
        );

        Ok(plan)
    }

    /// Estimate total inference latency
    fn estimate_inference_latency(
        &self,
        plan: &LayerAssignmentPlan,
        candidates: &[ElectionCandidate],
    ) -> u64 {
        let mut total_latency = 0u64;

        for assignment in plan.assignments.values() {
            // Find candidate
            if let Some(candidate) = candidates.iter().find(|c| c.node_id == assignment.node_id) {
                let layers_assigned = assignment.layer_end - assignment.layer_start + 1;

                // Estimate per-layer latency based on capability
                let per_layer_latency = match &assignment.device_capability {
                    DeviceCapability::CUDA { vram_gb, .. } => {
                        // High-end GPUs: ~5-10ms per layer
                        if *vram_gb >= 24 {
                            5
                        } else if *vram_gb >= 12 {
                            8
                        } else {
                            12
                        }
                    }
                    DeviceCapability::Metal { vram_gb } => {
                        // Apple Silicon: ~8-15ms per layer
                        if *vram_gb >= 32 {
                            8
                        } else if *vram_gb >= 16 {
                            12
                        } else {
                            15
                        }
                    }
                    DeviceCapability::CPU { cores, .. } => {
                        // CPUs: ~50-100ms per layer (much slower)
                        if *cores >= 16 {
                            50
                        } else if *cores >= 8 {
                            75
                        } else {
                            100
                        }
                    }
                };

                // Add network latency (avg 20ms per layer transfer)
                let network_latency = candidate.average_latency_ms.max(20);

                total_latency += (per_layer_latency * layers_assigned as u64) + network_latency;
            }
        }

        total_latency
    }

    /// Reassign layers after node failure
    pub fn reassign_after_failure(
        &self,
        current_plan: &LayerAssignmentPlan,
        failed_node: &str,
        available_candidates: Vec<ElectionCandidate>,
    ) -> Result<LayerAssignmentPlan> {
        info!("🔄 Reassigning layers after node failure: {}", failed_node);

        // Get the failed node's layer range
        let failed_assignment = current_plan
            .get_assignment(failed_node)
            .ok_or_else(|| anyhow!("Failed node not found in plan"))?;

        let failed_layers = (
            failed_assignment.layer_start,
            failed_assignment.layer_end,
        );

        info!(
            "📦 Reassigning layers {}-{} from failed node",
            failed_layers.0, failed_layers.1
        );

        // Create new plan with remaining nodes
        let mut new_plan = LayerAssignmentPlan::new(self.model_name.clone(), self.total_layers);

        // Copy assignments from healthy nodes
        for (node_id, assignment) in &current_plan.assignments {
            if node_id != failed_node {
                new_plan.add_assignment(assignment.clone());
            }
        }

        // Find best candidate to take over failed layers
        if available_candidates.is_empty() {
            return Err(anyhow!("No available nodes to reassign failed layers"));
        }

        let mut sorted = available_candidates.clone();
        sorted.sort_by(|a, b| b.election_score.cmp(&a.election_score));

        let replacement = &sorted[0];
        let layers_count = failed_layers.1 - failed_layers.0 + 1;

        let new_assignment = LayerAssignment {
            node_id: replacement.node_id.clone(),
            peer_id: replacement.peer_id.clone(),
            layer_start: failed_layers.0,
            layer_end: failed_layers.1,
            device_capability: replacement.capability.clone(),
            last_seen: chrono::Utc::now().timestamp(),
        };

        info!(
            "✅ Reassigned {} layers to node {}",
            layers_count, replacement.node_id
        );

        new_plan.add_assignment(new_assignment);
        new_plan.validate()?;

        Ok(new_plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candidate(node_id: &str, capability: DeviceCapability) -> ElectionCandidate {
        let mut candidate = ElectionCandidate {
            node_id: node_id.to_string(),
            peer_id: format!("peer-{}", node_id),
            capability,
            uptime_secs: 3600,
            inference_count: 100,
            average_latency_ms: 50,
            last_heartbeat: chrono::Utc::now().timestamp(),
            election_score: 0,
            stake_amount: 0,
        };
        candidate.calculate_election_score();
        candidate
    }

    #[test]
    fn test_layer_assignment_plan_validation() {
        let mut plan = LayerAssignmentPlan::new("test-model".to_string(), 10);

        // Add valid assignments
        plan.add_assignment(LayerAssignment {
            node_id: "node1".to_string(),
            peer_id: "peer1".to_string(),
            layer_start: 0,
            layer_end: 4,
            device_capability: DeviceCapability::CPU {
                cores: 8,
                ram_gb: 16,
            },
            last_seen: chrono::Utc::now().timestamp(),
        });

        plan.add_assignment(LayerAssignment {
            node_id: "node2".to_string(),
            peer_id: "peer2".to_string(),
            layer_start: 5,
            layer_end: 9,
            device_capability: DeviceCapability::CPU {
                cores: 4,
                ram_gb: 8,
            },
            last_seen: chrono::Utc::now().timestamp(),
        });

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_layer_assignment_coordinator() {
        let coordinator = LayerAssignmentCoordinator::for_mistral_7b();
        assert_eq!(coordinator.total_layers, 34);
        assert_eq!(coordinator.model_name, "mistral-7b-instruct-v0.3");
    }

    #[test]
    fn test_assign_layers_single_node() {
        let coordinator = LayerAssignmentCoordinator::new("test-model".to_string(), 10);

        let candidate = create_test_candidate(
            "node1",
            DeviceCapability::CUDA {
                vram_gb: 24,
                compute_capability: "8.6".to_string(),
            },
        );

        let plan = coordinator.assign_layers(vec![candidate]).unwrap();

        assert_eq!(plan.node_count(), 1);
        assert!(plan.validate().is_ok());

        let assignment = plan.get_assignment("node1").unwrap();
        assert_eq!(assignment.layer_start, 0);
        assert_eq!(assignment.layer_end, 9);
    }

    #[test]
    fn test_assign_layers_multiple_nodes() {
        let coordinator = LayerAssignmentCoordinator::new("test-model".to_string(), 34);

        let candidates = vec![
            create_test_candidate(
                "node1",
                DeviceCapability::CUDA {
                    vram_gb: 24,
                    compute_capability: "8.6".to_string(),
                },
            ),
            create_test_candidate(
                "node2",
                DeviceCapability::CUDA {
                    vram_gb: 12,
                    compute_capability: "7.5".to_string(),
                },
            ),
            create_test_candidate(
                "node3",
                DeviceCapability::CPU {
                    cores: 8,
                    ram_gb: 16,
                },
            ),
        ];

        let plan = coordinator.assign_layers(candidates).unwrap();

        // At least 2 nodes should be used (CUDA 24GB covers 24 layers, CUDA 12GB covers remaining 10)
        assert!(plan.node_count() >= 2);
        assert!(plan.validate().is_ok());

        // All 34 layers should be covered
        let total_assigned: usize = plan.assignments.values()
            .map(|a| a.layer_end - a.layer_start + 1)
            .sum();
        assert_eq!(total_assigned, 34);
    }

    #[test]
    fn test_reassign_after_failure() {
        let coordinator = LayerAssignmentCoordinator::new("test-model".to_string(), 20);

        let candidates = vec![
            create_test_candidate(
                "node1",
                DeviceCapability::CUDA {
                    vram_gb: 12,
                    compute_capability: "7.5".to_string(),
                },
            ),
            create_test_candidate(
                "node2",
                DeviceCapability::CUDA {
                    vram_gb: 12,
                    compute_capability: "7.5".to_string(),
                },
            ),
        ];

        let initial_plan = coordinator.assign_layers(candidates.clone()).unwrap();
        assert!(initial_plan.validate().is_ok());

        // Simulate node1 failure
        let replacement = create_test_candidate(
            "node3",
            DeviceCapability::CUDA {
                vram_gb: 24,
                compute_capability: "8.6".to_string(),
            },
        );

        let new_plan = coordinator
            .reassign_after_failure(&initial_plan, "node1", vec![replacement])
            .unwrap();

        assert!(new_plan.validate().is_ok());
        assert!(new_plan.get_assignment("node1").is_none());
        assert!(new_plan.get_assignment("node3").is_some());
    }
}
