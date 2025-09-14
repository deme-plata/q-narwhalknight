//! State management for DAGKnight
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time in milliseconds
    pub cpu_time: u64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// GPU time in milliseconds (if used)
    pub gpu_time: Option<u64>,
}

impl ResourceUsage {
    /// Create a minimal resource usage entry for cached results
    pub fn minimal() -> Self {
        Self {
            cpu_time: 1,
            memory_used: 1024,
            gpu_time: None,
        }
    }
    
    /// Calculate resource cost based on usage
    pub fn calculate_cost(&self) -> u64 {
        // Base cost from CPU
        let cpu_cost = self.cpu_time * 1; // 1 token per ms of CPU time
        
        // Memory cost
        let memory_cost = (self.memory_used / (1024 * 1024)) * 10; // 10 tokens per MB
        
        // GPU cost if used
        let gpu_cost = self.gpu_time.map(|time| time * 5).unwrap_or(0); // 5 tokens per ms of GPU time
        
        cpu_cost + memory_cost + gpu_cost
    }
}

/// Resource ledger for tracking contributions
pub struct ResourceLedger {
    /// Tracks resource contributions per node
    contributions: RwLock<HashMap<[u8; 32], ResourcePool>>,
}

/// Resource pool for a single node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourcePool {
    /// Total CPU time provided (ms)
    pub total_cpu: u64,
    /// Total memory provided (MB)
    pub total_memory: u64,
    /// Total GPU time provided (ms)
    pub total_gpu: u64,
    /// Pending rewards (tokens)
    pub pending_rewards: u64,
}

impl ResourceLedger {
    /// Create a new resource ledger
    pub fn new() -> Self {
        Self {
            contributions: RwLock::new(HashMap::new()),
        }
    }
    
    /// Update resource usage for a node
    pub async fn update_resources(&self, node: [u8; 32], usage: &ResourceUsage) {
        let mut ledger = self.contributions.write().await;
        let pool = ledger.entry(node).or_insert(ResourcePool::default());
        
        pool.total_cpu += usage.cpu_time;
        pool.total_memory += usage.memory_used / (1024 * 1024); // Convert to MB
        pool.total_gpu += usage.gpu_time.unwrap_or(0);
        
        // Calculate rewards
        let reward = usage.calculate_cost();
        pool.pending_rewards += reward;
    }
    
    /// Get resource pool for a node
    pub async fn get_resource_pool(&self, node: &[u8; 32]) -> Option<ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.get(node).cloned()
    }
    
    /// Get all resource pools
    pub async fn get_all_resource_pools(&self) -> HashMap<[u8; 32],
    ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.clone()
    }
    
    /// Claim rewards for a node
    pub async fn claim_rewards(&self, node: &[u8; 32]) -> u64 {
        let mut ledger = self.contributions.write().await;
        
        if let Some(pool) = ledger.get_mut(node) {
            let rewards = pool.pending_rewards;
            pool.pending_rewards = 0;
            rewards
        } else {
            0
        }
    }
    
    /// Get total resource usage across all nodes
    pub async fn get_total_resource_usage(&self) -> ResourcePool {
        let ledger = self.contributions.read().await;
        
        let mut total = ResourcePool::default();
        for pool in ledger.values() {
            total.total_cpu += pool.total_cpu;
            total.total_memory += pool.total_memory;
            total.total_gpu += pool.total_gpu;
            total.pending_rewards += pool.pending_rewards;
        }
        
        total
    }
}

/// State database for DAGKnight
#[derive(Debug)]
pub struct StateDB {
#[derive(Debug)]
#[derive(Debug)]
    /// Resource ledger
    pub resource_ledger: ResourceLedger,
    // Other state components would go here
}

impl StateDB {
    /// Create a new state database
    pub fn new() -> Self {
        Self {
            resource_ledger: ResourceLedger::new(),
        }
    }
    
    /// Update resource ledger
    pub async fn update_resource_ledger(
        &self,
        node: [u8; 32],
        usage: &ResourceUsage
    ) {
        self.resource_ledger.update_resources(node, usage).await;
    }
}

impl Default for StateDB {
    fn default() -> Self {
        Self::new()
    }
}
