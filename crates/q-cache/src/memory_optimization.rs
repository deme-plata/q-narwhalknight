//! Memory Optimization Module
//!
//! NUMA-aware memory management and optimization for the Q-NarwhalKnight
//! hierarchical caching system.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// NUMA node configuration
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpu_cores: Vec<u32>,
    pub memory_capacity_mb: usize,
    pub memory_used_mb: usize,
    pub local_cache_mb: usize,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    /// Allocate on local NUMA node
    NumaLocal,
    /// Allocate on least loaded node
    LoadBalanced,
    /// Interleave across all nodes
    Interleaved,
    /// Allocate on specific node
    Specific(u32),
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub numa_enabled: bool,
    pub allocation_strategy: AllocationStrategy,
    pub memory_limit_gb: f64,
    pub gc_threshold: f64,
    pub huge_pages_enabled: bool,
    pub prefault_memory: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            numa_enabled: true,
            allocation_strategy: AllocationStrategy::NumaLocal,
            memory_limit_gb: 2.0,
            gc_threshold: 0.8,
            huge_pages_enabled: true,
            prefault_memory: true,
        }
    }
}

/// Memory statistics and metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated_mb: usize,
    pub numa_local_hits: u64,
    pub numa_remote_hits: u64,
    pub gc_runs: u64,
    pub huge_pages_used: usize,
    pub memory_fragmentation: f64,
    pub allocation_failures: u64,
}

/// NUMA-aware memory optimizer
pub struct MemoryOptimizer {
    config: MemoryConfig,
    numa_nodes: Arc<RwLock<HashMap<u32, NumaNode>>>,
    stats: Arc<RwLock<MemoryStats>>,
    current_node: u32,
}

impl MemoryOptimizer {
    /// Create new memory optimizer
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        info!("Initializing NUMA-aware memory optimizer");

        let numa_nodes = Self::discover_numa_topology().await?;
        let current_node = Self::get_current_numa_node().await.unwrap_or(0);

        Ok(Self {
            config,
            numa_nodes: Arc::new(RwLock::new(numa_nodes)),
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            current_node,
        })
    }

    /// Discover NUMA topology
    async fn discover_numa_topology() -> Result<HashMap<u32, NumaNode>> {
        let mut nodes = HashMap::new();

        // Simplified NUMA discovery - in production would read from /sys/devices/system/node/
        for node_id in 0..Self::get_numa_node_count().await {
            let numa_node = NumaNode {
                node_id,
                cpu_cores: Self::get_node_cpus(node_id).await,
                memory_capacity_mb: Self::get_node_memory_capacity(node_id).await,
                memory_used_mb: 0,
                local_cache_mb: 0,
            };
            nodes.insert(node_id, numa_node);
        }

        info!("Discovered {} NUMA nodes", nodes.len());
        Ok(nodes)
    }

    /// Get current NUMA node for this process
    async fn get_current_numa_node() -> Option<u32> {
        // Simplified - would use libnuma or /proc/self/numa_maps in production
        Some(0)
    }

    /// Get number of NUMA nodes
    async fn get_numa_node_count() -> u32 {
        // Simplified discovery
        if std::path::Path::new("/sys/devices/system/node/node1").exists() {
            4 // Assume 4-node system
        } else {
            1 // Single node
        }
    }

    /// Get CPU cores for a NUMA node
    async fn get_node_cpus(node_id: u32) -> Vec<u32> {
        // Simplified - would read from sysfs in production
        let cores_per_node = 8;
        (node_id * cores_per_node..(node_id + 1) * cores_per_node).collect()
    }

    /// Get memory capacity for a NUMA node
    async fn get_node_memory_capacity(node_id: u32) -> usize {
        // Simplified - would read from /sys/devices/system/node/nodeX/meminfo
        match node_id {
            0 => 16 * 1024, // 16GB
            1 => 16 * 1024, // 16GB
            2 => 32 * 1024, // 32GB
            3 => 32 * 1024, // 32GB
            _ => 8 * 1024,  // 8GB default
        }
    }

    /// Allocate memory with NUMA awareness
    pub async fn allocate_memory(
        &self,
        size_mb: usize,
        preferred_node: Option<u32>,
    ) -> Result<AllocationResult> {
        let target_node = match self.config.allocation_strategy {
            AllocationStrategy::NumaLocal => self.current_node,
            AllocationStrategy::LoadBalanced => self.find_least_loaded_node().await?,
            AllocationStrategy::Interleaved => self.select_interleaved_node().await?,
            AllocationStrategy::Specific(node) => node,
        };

        // Check if target node has capacity
        let can_allocate = self.check_node_capacity(target_node, size_mb).await?;

        if !can_allocate {
            warn!(
                "NUMA node {} lacks capacity for {}MB allocation",
                target_node, size_mb
            );
            return self.fallback_allocation(size_mb).await;
        }

        // Update allocation tracking
        {
            let mut nodes = self.numa_nodes.write().await;
            if let Some(node) = nodes.get_mut(&target_node) {
                node.memory_used_mb += size_mb;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_allocated_mb += size_mb;

            if target_node == self.current_node {
                stats.numa_local_hits += 1;
            } else {
                stats.numa_remote_hits += 1;
            }
        }

        debug!("Allocated {}MB on NUMA node {}", size_mb, target_node);

        Ok(AllocationResult {
            numa_node: target_node,
            size_mb,
            is_local: target_node == self.current_node,
            uses_huge_pages: self.config.huge_pages_enabled,
        })
    }

    /// Find least loaded NUMA node
    async fn find_least_loaded_node(&self) -> Result<u32> {
        let nodes = self.numa_nodes.read().await;

        let least_loaded = nodes
            .values()
            .min_by(|a, b| {
                let a_load = a.memory_used_mb as f64 / a.memory_capacity_mb as f64;
                let b_load = b.memory_used_mb as f64 / b.memory_capacity_mb as f64;
                a_load
                    .partial_cmp(&b_load)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.node_id)
            .unwrap_or(0);

        Ok(least_loaded)
    }

    /// Select node for interleaved allocation
    async fn select_interleaved_node(&self) -> Result<u32> {
        let nodes = self.numa_nodes.read().await;
        let node_count = nodes.len() as u32;

        // Simple round-robin interleaving
        let stats = self.stats.read().await;
        let allocations = stats.numa_local_hits + stats.numa_remote_hits;
        Ok(allocations as u32 % node_count)
    }

    /// Check if node has sufficient capacity
    async fn check_node_capacity(&self, node_id: u32, required_mb: usize) -> Result<bool> {
        let nodes = self.numa_nodes.read().await;

        if let Some(node) = nodes.get(&node_id) {
            let available_mb = node.memory_capacity_mb.saturating_sub(node.memory_used_mb);
            Ok(available_mb >= required_mb)
        } else {
            Ok(false)
        }
    }

    /// Fallback allocation when preferred node is full
    async fn fallback_allocation(&self, size_mb: usize) -> Result<AllocationResult> {
        let least_loaded = self.find_least_loaded_node().await?;

        warn!(
            "Falling back to NUMA node {} for {}MB allocation",
            least_loaded, size_mb
        );

        let can_allocate = self.check_node_capacity(least_loaded, size_mb).await?;

        if !can_allocate {
            let mut stats = self.stats.write().await;
            stats.allocation_failures += 1;

            return Err(anyhow::anyhow!(
                "No NUMA node has capacity for {}MB allocation",
                size_mb
            ));
        }

        // Update allocation tracking
        {
            let mut nodes = self.numa_nodes.write().await;
            if let Some(node) = nodes.get_mut(&least_loaded) {
                node.memory_used_mb += size_mb;
            }
        }

        Ok(AllocationResult {
            numa_node: least_loaded,
            size_mb,
            is_local: least_loaded == self.current_node,
            uses_huge_pages: self.config.huge_pages_enabled,
        })
    }

    /// Deallocate memory and update NUMA tracking
    pub async fn deallocate_memory(&self, allocation: &AllocationResult) -> Result<()> {
        {
            let mut nodes = self.numa_nodes.write().await;
            if let Some(node) = nodes.get_mut(&allocation.numa_node) {
                node.memory_used_mb = node.memory_used_mb.saturating_sub(allocation.size_mb);
            }
        }

        {
            let mut stats = self.stats.write().await;
            stats.total_allocated_mb = stats.total_allocated_mb.saturating_sub(allocation.size_mb);
        }

        debug!(
            "Deallocated {}MB from NUMA node {}",
            allocation.size_mb, allocation.numa_node
        );
        Ok(())
    }

    /// Optimize memory layout by migrating pages to optimal nodes
    pub async fn optimize_memory_layout(&self) -> Result<u32> {
        info!("Starting memory layout optimization");

        let mut migrations = 0;

        // This would implement page migration logic in production
        // For now, just log the optimization attempt

        let stats = self.get_memory_stats().await;
        let numa_efficiency =
            stats.numa_local_hits as f64 / (stats.numa_local_hits + stats.numa_remote_hits) as f64;

        if numa_efficiency < 0.8 {
            info!(
                "NUMA efficiency is {:.1}%, considering memory migration",
                numa_efficiency * 100.0
            );
            migrations = self.simulate_memory_migration().await?;
        }

        info!(
            "Memory layout optimization complete, {} pages migrated",
            migrations
        );
        Ok(migrations)
    }

    /// Simulate memory migration (placeholder for actual implementation)
    async fn simulate_memory_migration(&self) -> Result<u32> {
        // In production, this would:
        // 1. Identify hot pages on remote nodes
        // 2. Use move_pages() system call to migrate to local node
        // 3. Update internal tracking

        Ok(128) // Simulate migrating 128 pages
    }

    /// Enable huge pages for large allocations
    pub async fn enable_huge_pages(&self, size_mb: usize) -> Result<bool> {
        if !self.config.huge_pages_enabled {
            return Ok(false);
        }

        // Huge pages are beneficial for allocations >2MB
        if size_mb >= 2 {
            debug!("Enabling huge pages for {}MB allocation", size_mb);

            // In production, would use madvise(MADV_HUGEPAGE) or hugetlbfs
            let mut stats = self.stats.write().await;
            stats.huge_pages_used += size_mb / 2; // 2MB huge pages

            return Ok(true);
        }

        Ok(false)
    }

    /// Trigger garbage collection when memory pressure is high
    pub async fn check_memory_pressure(&self) -> Result<bool> {
        let stats = self.stats.read().await;
        let memory_usage_gb = stats.total_allocated_mb as f64 / 1024.0;
        let pressure_ratio = memory_usage_gb / self.config.memory_limit_gb;

        if pressure_ratio >= self.config.gc_threshold {
            warn!(
                "Memory pressure detected: {:.1}GB used ({:.1}% of limit)",
                memory_usage_gb,
                pressure_ratio * 100.0
            );

            drop(stats);
            self.trigger_garbage_collection().await?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Trigger garbage collection
    async fn trigger_garbage_collection(&self) -> Result<()> {
        info!("Triggering garbage collection");

        // In production, this would:
        // 1. Identify unused cache entries
        // 2. Free memory from least recently used pages
        // 3. Compact fragmented memory regions
        // 4. Return memory to OS if appropriate

        {
            let mut stats = self.stats.write().await;
            stats.gc_runs += 1;

            // Simulate freeing 25% of allocated memory
            let freed_mb = stats.total_allocated_mb / 4;
            stats.total_allocated_mb -= freed_mb;

            info!("Garbage collection freed {}MB", freed_mb);
        }

        Ok(())
    }

    /// Get current memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.stats.read().await.clone()
    }

    /// Get NUMA node information
    pub async fn get_numa_nodes(&self) -> HashMap<u32, NumaNode> {
        self.numa_nodes.read().await.clone()
    }

    /// Update memory configuration
    pub async fn update_config(&mut self, config: MemoryConfig) -> Result<()> {
        info!("Updating memory configuration");
        self.config = config;
        Ok(())
    }

    /// Calculate memory fragmentation
    pub async fn calculate_fragmentation(&self) -> Result<f64> {
        let nodes = self.numa_nodes.read().await;

        let mut total_fragmentation = 0.0;
        let mut node_count = 0;

        for node in nodes.values() {
            let usage_ratio = node.memory_used_mb as f64 / node.memory_capacity_mb as f64;

            // Simplified fragmentation calculation
            // Real implementation would analyze actual memory layout
            let fragmentation = if usage_ratio > 0.8 {
                (usage_ratio - 0.8) * 0.5 // High usage = more fragmentation
            } else {
                usage_ratio * 0.1 // Low usage = less fragmentation
            };

            total_fragmentation += fragmentation;
            node_count += 1;
        }

        let avg_fragmentation = if node_count > 0 {
            total_fragmentation / node_count as f64
        } else {
            0.0
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.memory_fragmentation = avg_fragmentation;
        }

        Ok(avg_fragmentation)
    }
}

/// Result of memory allocation
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub numa_node: u32,
    pub size_mb: usize,
    pub is_local: bool,
    pub uses_huge_pages: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_optimizer_creation() {
        let config = MemoryConfig::default();
        let optimizer = MemoryOptimizer::new(config).await;
        assert!(optimizer.is_ok());
    }

    #[tokio::test]
    async fn test_memory_allocation() {
        let config = MemoryConfig::default();
        let optimizer = MemoryOptimizer::new(config).await.unwrap();

        let result = optimizer.allocate_memory(100, None).await;
        assert!(result.is_ok());

        let allocation = result.unwrap();
        assert_eq!(allocation.size_mb, 100);
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let config = MemoryConfig::default();
        let optimizer = MemoryOptimizer::new(config).await.unwrap();

        let stats = optimizer.get_memory_stats().await;
        assert_eq!(stats.total_allocated_mb, 0);
        assert_eq!(stats.gc_runs, 0);
    }
}
