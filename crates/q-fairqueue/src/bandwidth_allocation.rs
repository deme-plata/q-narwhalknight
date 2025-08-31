/// Bandwidth allocation strategies for quantum fair queueing

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tracing::debug;

use q_types::NodeId;

/// Bandwidth allocation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Equal bandwidth for all nodes
    Equal,
    
    /// Proportional to node stake/reputation
    Proportional,
    
    /// Max-min fairness
    MaxMinFair,
    
    /// Proportional fair allocation
    ProportionalFair,
    
    /// Quantum-enhanced fair allocation
    QuantumFair,
}

/// Bandwidth allocator
pub struct BandwidthAllocator {
    strategy: AllocationStrategy,
    node_allocations: HashMap<NodeId, u64>,
    total_bandwidth: u64,
    node_weights: HashMap<NodeId, f64>,
}

impl BandwidthAllocator {
    /// Create new bandwidth allocator
    pub fn new(strategy: AllocationStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            node_allocations: HashMap::new(),
            total_bandwidth: 1_000_000, // 1 Mbps default
            node_weights: HashMap::new(),
        })
    }
    
    /// Update bandwidth allocations
    pub async fn update_allocations(&mut self, node_demands: HashMap<NodeId, u64>) -> Result<()> {
        debug!("Updating bandwidth allocations for {} nodes", node_demands.len());
        
        match self.strategy {
            AllocationStrategy::Equal => {
                self.allocate_equal(&node_demands).await?;
            },
            AllocationStrategy::Proportional => {
                self.allocate_proportional(&node_demands).await?;
            },
            AllocationStrategy::MaxMinFair => {
                self.allocate_max_min_fair(&node_demands).await?;
            },
            AllocationStrategy::ProportionalFair => {
                self.allocate_proportional_fair(&node_demands).await?;
            },
            AllocationStrategy::QuantumFair => {
                self.allocate_quantum_fair(&node_demands).await?;
            },
        }
        
        Ok(())
    }
    
    /// Equal allocation
    async fn allocate_equal(&mut self, node_demands: &HashMap<NodeId, u64>) -> Result<()> {
        if node_demands.is_empty() {
            return Ok(());
        }
        
        let per_node = self.total_bandwidth / node_demands.len() as u64;
        
        for &node_id in node_demands.keys() {
            self.node_allocations.insert(node_id, per_node);
        }
        
        debug!("Equal allocation: {} bytes/s per node", per_node);
        Ok(())
    }
    
    /// Proportional allocation based on weights
    async fn allocate_proportional(&mut self, node_demands: &HashMap<NodeId, u64>) -> Result<()> {
        let total_weight: f64 = node_demands.keys()
            .map(|node_id| *self.node_weights.get(node_id).unwrap_or(&1.0))
            .sum();
        
        if total_weight <= 0.0 {
            return self.allocate_equal(node_demands).await;
        }
        
        for (&node_id, _) in node_demands {
            let weight = *self.node_weights.get(&node_id).unwrap_or(&1.0);
            let allocation = ((weight / total_weight) * self.total_bandwidth as f64) as u64;
            self.node_allocations.insert(node_id, allocation);
        }
        
        debug!("Proportional allocation completed (total_weight: {})", total_weight);
        Ok(())
    }
    
    /// Max-min fair allocation
    async fn allocate_max_min_fair(&mut self, node_demands: &HashMap<NodeId, u64>) -> Result<()> {
        let mut remaining_bandwidth = self.total_bandwidth;
        let mut remaining_nodes: Vec<NodeId> = node_demands.keys().cloned().collect();
        let mut allocations = HashMap::new();
        
        // Sort nodes by demand (ascending)
        remaining_nodes.sort_by_key(|node_id| node_demands[node_id]);
        
        while !remaining_nodes.is_empty() && remaining_bandwidth > 0 {
            let fair_share = remaining_bandwidth / remaining_nodes.len() as u64;
            let mut to_remove = Vec::new();
            
            for &node_id in &remaining_nodes {
                let demand = node_demands[&node_id];
                if demand <= fair_share {
                    // Node's demand can be fully satisfied
                    allocations.insert(node_id, demand);
                    remaining_bandwidth -= demand;
                    to_remove.push(node_id);
                }
            }
            
            // Remove satisfied nodes
            for node_id in to_remove {
                remaining_nodes.retain(|&x| x != node_id);
            }
            
            // If no nodes were satisfied, allocate fair share to all
            if remaining_nodes.len() == node_demands.len() {
                for &node_id in &remaining_nodes {
                    allocations.insert(node_id, fair_share);
                }
                break;
            }
        }
        
        self.node_allocations = allocations;
        debug!("Max-min fair allocation completed");
        Ok(())
    }
    
    /// Proportional fair allocation
    async fn allocate_proportional_fair(&mut self, node_demands: &HashMap<NodeId, u64>) -> Result<()> {
        // Proportional fair maximizes sum of log(allocations)
        let mut allocations = HashMap::new();
        let num_nodes = node_demands.len();
        
        if num_nodes == 0 {
            return Ok(());
        }
        
        // Start with equal allocation
        let initial_allocation = self.total_bandwidth / num_nodes as u64;
        for &node_id in node_demands.keys() {
            allocations.insert(node_id, initial_allocation);
        }
        
        // Iteratively improve allocations
        for _ in 0..10 { // Limit iterations
            let mut improved = false;
            
            for (&node_a, &demand_a) in node_demands {
                for (&node_b, &demand_b) in node_demands {
                    if node_a == node_b { continue; }
                    
                    let alloc_a = allocations[&node_a];
                    let alloc_b = allocations[&node_b];
                    
                    // Check if transferring bandwidth improves proportional fairness
                    if alloc_a > demand_a && alloc_b < demand_b {
                        let transfer = (alloc_a - demand_a).min(demand_b - alloc_b).min(1024);
                        if transfer > 0 {
                            *allocations.get_mut(&node_a).unwrap() -= transfer;
                            *allocations.get_mut(&node_b).unwrap() += transfer;
                            improved = true;
                        }
                    }
                }
            }
            
            if !improved { break; }
        }
        
        self.node_allocations = allocations;
        debug!("Proportional fair allocation completed");
        Ok(())
    }
    
    /// Quantum-enhanced fair allocation
    async fn allocate_quantum_fair(&mut self, node_demands: &HashMap<NodeId, u64>) -> Result<()> {
        // For now, fall back to proportional fair with quantum adjustments
        self.allocate_proportional_fair(node_demands).await?;
        
        // Add small quantum randomness to prevent predictable patterns
        // This would use QRNG in a real implementation
        for allocation in self.node_allocations.values_mut() {
            // Add ±1% random adjustment
            let adjustment = (*allocation as f64 * 0.01) as u64;
            let random_adjustment = (allocation.wrapping_mul(17) % (2 * adjustment + 1)) as i64 - adjustment as i64;
            *allocation = (*allocation as i64 + random_adjustment).max(0) as u64;
        }
        
        debug!("Quantum fair allocation completed with randomness adjustments");
        Ok(())
    }
    
    /// Get allocation for specific node
    pub fn get_allocation(&self, node_id: NodeId) -> u64 {
        *self.node_allocations.get(&node_id).unwrap_or(&0)
    }
    
    /// Set node weights for proportional allocation
    pub fn set_node_weights(&mut self, weights: HashMap<NodeId, f64>) {
        self.node_weights = weights;
    }
    
    /// Set total bandwidth
    pub fn set_total_bandwidth(&mut self, bandwidth: u64) {
        self.total_bandwidth = bandwidth;
    }
    
    /// Get current allocations
    pub fn get_all_allocations(&self) -> &HashMap<NodeId, u64> {
        &self.node_allocations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_allocator_creation() {
        let allocator = BandwidthAllocator::new(AllocationStrategy::Equal).unwrap();
        assert_eq!(allocator.total_bandwidth, 1_000_000);
    }

    #[tokio::test]
    async fn test_equal_allocation() {
        let mut allocator = BandwidthAllocator::new(AllocationStrategy::Equal).unwrap();
        
        let mut demands = HashMap::new();
        demands.insert([1u8; 32], 500_000);
        demands.insert([2u8; 32], 800_000);
        
        allocator.update_allocations(demands).await.unwrap();
        
        assert_eq!(allocator.get_allocation([1u8; 32]), 500_000);
        assert_eq!(allocator.get_allocation([2u8; 32]), 500_000);
    }

    #[tokio::test]
    async fn test_proportional_allocation() {
        let mut allocator = BandwidthAllocator::new(AllocationStrategy::Proportional).unwrap();
        
        let mut weights = HashMap::new();
        weights.insert([1u8; 32], 1.0);
        weights.insert([2u8; 32], 2.0);
        allocator.set_node_weights(weights);
        
        let mut demands = HashMap::new();
        demands.insert([1u8; 32], 1_000_000);
        demands.insert([2u8; 32], 1_000_000);
        
        allocator.update_allocations(demands).await.unwrap();
        
        let alloc1 = allocator.get_allocation([1u8; 32]);
        let alloc2 = allocator.get_allocation([2u8; 32]);
        
        // Node 2 should get twice the allocation of node 1
        assert!(alloc2 > alloc1);
        assert!((alloc2 as f64 / alloc1 as f64 - 2.0).abs() < 0.1);
    }
}