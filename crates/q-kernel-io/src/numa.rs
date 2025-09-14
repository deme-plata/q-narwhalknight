// NUMA-Aware Memory Management
// CPU topology detection and NUMA-optimized memory allocation

use anyhow::Result;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub cpu_cores: BTreeSet<usize>,
    pub memory_size_mb: u64,
    pub memory_free_mb: u64,
    pub distance_map: HashMap<usize, u32>, // Distance to other NUMA nodes
}

impl NumaNode {
    /// Create new NUMA node
    pub fn new(node_id: usize) -> Self {
        Self {
            node_id,
            cpu_cores: BTreeSet::new(),
            memory_size_mb: 0,
            memory_free_mb: 0,
            distance_map: HashMap::new(),
        }
    }

    /// Check if this node is local to the current thread
    pub fn is_local(&self) -> bool {
        // Get current CPU
        if let Ok(cpu_id) = Self::get_current_cpu() {
            self.cpu_cores.contains(&cpu_id)
        } else {
            false
        }
    }

    /// Get distance to another NUMA node
    pub fn distance_to(&self, other_node: usize) -> u32 {
        self.distance_map.get(&other_node).copied().unwrap_or(255)
    }

    /// Get current CPU ID
    #[cfg(target_os = "linux")]
    fn get_current_cpu() -> Result<usize> {
        use std::fs;

        // Read current CPU from /proc/self/stat
        let stat = fs::read_to_string("/proc/self/stat")?;
        let fields: Vec<&str> = stat.split_whitespace().collect();

        // CPU is field 39 (0-indexed)
        if fields.len() > 39 {
            Ok(fields[39].parse()?)
        } else {
            Err(anyhow::anyhow!("Unable to parse CPU from /proc/self/stat"))
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn get_current_cpu() -> Result<usize> {
        // Fallback for non-Linux systems
        Ok(0)
    }
}

/// Complete NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    nodes: HashMap<usize, NumaNode>,
    total_nodes: usize,
    total_cpus: usize,
    total_memory_mb: u64,
}

impl NumaTopology {
    /// Detect NUMA topology from system
    pub async fn detect() -> Result<Self> {
        info!("Detecting NUMA topology");

        let mut topology = Self {
            nodes: HashMap::new(),
            total_nodes: 0,
            total_cpus: 0,
            total_memory_mb: 0,
        };

        #[cfg(target_os = "linux")]
        {
            topology.detect_linux_numa().await?;
        }

        #[cfg(not(target_os = "linux"))]
        {
            topology.detect_generic_numa().await?;
        }

        info!(
            "NUMA topology detected: {} nodes, {} CPUs, {} MB memory",
            topology.total_nodes, topology.total_cpus, topology.total_memory_mb
        );

        Ok(topology)
    }

    #[cfg(target_os = "linux")]
    async fn detect_linux_numa(&mut self) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");
        if !numa_path.exists() {
            // Single NUMA node system
            let mut node = NumaNode::new(0);

            // Get all CPUs
            let cpu_count = num_cpus::get();
            for cpu_id in 0..cpu_count {
                node.cpu_cores.insert(cpu_id);
            }

            // Get memory info
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                node.memory_size_mb = Self::parse_memory_from_meminfo(&meminfo, "MemTotal")?;
                node.memory_free_mb = Self::parse_memory_from_meminfo(&meminfo, "MemAvailable")?;
                self.total_memory_mb = node.memory_size_mb;
            }

            self.nodes.insert(0, node);
            self.total_nodes = 1;
            self.total_cpus = cpu_count;

            return Ok(());
        }

        // Multi-NUMA system - enumerate nodes
        for entry in fs::read_dir(numa_path)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("node") {
                if let Ok(node_id) = name_str[4..].parse::<usize>() {
                    let mut node = NumaNode::new(node_id);
                    let node_path = entry.path();

                    // Read CPU list
                    let cpulist_path = node_path.join("cpulist");
                    if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                        node.cpu_cores = Self::parse_cpu_list(&cpulist)?;
                        self.total_cpus += node.cpu_cores.len();
                    }

                    // Read memory info
                    let meminfo_path = node_path.join("meminfo");
                    if let Ok(meminfo) = fs::read_to_string(&meminfo_path) {
                        node.memory_size_mb =
                            Self::parse_memory_from_meminfo(&meminfo, "MemTotal")?;
                        node.memory_free_mb = Self::parse_memory_from_meminfo(&meminfo, "MemFree")?;
                        self.total_memory_mb += node.memory_size_mb;
                    }

                    // Read distance map
                    let distance_path = node_path.join("distance");
                    if let Ok(distance_data) = fs::read_to_string(&distance_path) {
                        node.distance_map = Self::parse_distance_map(&distance_data)?;
                    }

                    self.nodes.insert(node_id, node);
                    self.total_nodes += 1;
                }
            }
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    async fn detect_generic_numa(&mut self) -> Result<()> {
        // Fallback for non-Linux systems - assume single NUMA node
        let mut node = NumaNode::new(0);

        let cpu_count = num_cpus::get();
        for cpu_id in 0..cpu_count {
            node.cpu_cores.insert(cpu_id);
        }

        // Estimate memory (this is very rough)
        node.memory_size_mb = 8192; // Default 8GB assumption
        node.memory_free_mb = 4096; // Assume half available

        self.nodes.insert(0, node);
        self.total_nodes = 1;
        self.total_cpus = cpu_count;
        self.total_memory_mb = 8192;

        Ok(())
    }

    fn parse_cpu_list(cpulist: &str) -> Result<BTreeSet<usize>> {
        let mut cpus = BTreeSet::new();

        for range in cpulist.trim().split(',') {
            if range.contains('-') {
                let parts: Vec<&str> = range.split('-').collect();
                if parts.len() == 2 {
                    let start: usize = parts[0].parse()?;
                    let end: usize = parts[1].parse()?;
                    for cpu in start..=end {
                        cpus.insert(cpu);
                    }
                }
            } else {
                cpus.insert(range.parse()?);
            }
        }

        Ok(cpus)
    }

    fn parse_memory_from_meminfo(meminfo: &str, field: &str) -> Result<u64> {
        for line in meminfo.lines() {
            if line.starts_with(field) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb_value: u64 = parts[1].parse()?;
                    return Ok(kb_value / 1024); // Convert KB to MB
                }
            }
        }
        Ok(0)
    }

    fn parse_distance_map(distance_data: &str) -> Result<HashMap<usize, u32>> {
        let mut distance_map = HashMap::new();

        for (node_id, distance_str) in distance_data.trim().split_whitespace().enumerate() {
            let distance: u32 = distance_str.parse()?;
            distance_map.insert(node_id, distance);
        }

        Ok(distance_map)
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&NumaNode> {
        self.nodes.get(&node_id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<usize, NumaNode> {
        &self.nodes
    }

    /// Get total number of NUMA nodes
    pub fn node_count(&self) -> usize {
        self.total_nodes
    }

    /// Get optimal NUMA node for current thread
    pub fn get_local_node(&self) -> Option<&NumaNode> {
        self.nodes.values().find(|node| node.is_local())
    }

    /// Find node with most available memory
    pub fn find_node_with_most_memory(&self) -> Option<&NumaNode> {
        self.nodes.values().max_by_key(|node| node.memory_free_mb)
    }

    /// Get nodes sorted by distance from given node
    pub fn get_nodes_by_distance(&self, from_node: usize) -> Vec<&NumaNode> {
        let mut nodes: Vec<&NumaNode> = self.nodes.values().collect();

        if let Some(source_node) = self.nodes.get(&from_node) {
            nodes.sort_by_key(|node| source_node.distance_to(node.node_id));
        }

        nodes
    }
}

/// NUMA performance metrics
#[derive(Debug, Clone, Default)]
pub struct NumaMetrics {
    pub total_memory: u64,
    pub available_memory: u64,
    pub local_allocations: u64,
    pub remote_allocations: u64,
    pub memory_bandwidth_mb_per_sec: f64,
    pub numa_misses: u64,
    pub cpu_migrations: u64,
}

/// NUMA-aware memory manager
#[derive(Debug)]
pub struct NumaManager {
    topology: NumaTopology,
    metrics: Arc<RwLock<NumaMetrics>>,
    allocation_policy: NumaAllocationPolicy,
}

/// NUMA memory allocation policies
#[derive(Debug, Clone)]
pub enum NumaAllocationPolicy {
    /// Prefer local NUMA node
    LocalPreferred,
    /// Strictly allocate on local NUMA node
    LocalOnly,
    /// Interleave across all nodes
    Interleave,
    /// Bind to specific node
    BindToNode(usize),
}

impl NumaManager {
    /// Create new NUMA manager
    pub async fn new(preferred_nodes: usize) -> Result<Self> {
        let topology = NumaTopology::detect().await?;

        // Choose allocation policy based on topology
        let allocation_policy = if topology.node_count() > 1 {
            NumaAllocationPolicy::LocalPreferred
        } else {
            NumaAllocationPolicy::BindToNode(0)
        };

        info!(
            "NUMA manager initialized with {} nodes",
            topology.node_count()
        );

        Ok(Self {
            topology,
            metrics: Arc::new(RwLock::new(NumaMetrics::default())),
            allocation_policy,
        })
    }

    /// Get NUMA topology
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Get number of NUMA nodes
    pub fn node_count(&self) -> usize {
        self.topology.node_count()
    }

    /// Get optimal node for allocation
    pub fn get_optimal_node(&self) -> Option<usize> {
        match &self.allocation_policy {
            NumaAllocationPolicy::LocalPreferred | NumaAllocationPolicy::LocalOnly => {
                self.topology.get_local_node().map(|node| node.node_id)
            }
            NumaAllocationPolicy::Interleave => {
                // Simple round-robin selection
                Some(0) // Simplified
            }
            NumaAllocationPolicy::BindToNode(node_id) => Some(*node_id),
        }
    }

    /// Allocate memory on specific NUMA node
    pub async fn allocate_on_node(&self, size: usize, node_id: usize) -> Result<*mut u8> {
        #[cfg(target_os = "linux")]
        {
            self.allocate_linux_numa(size, node_id).await
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback to regular allocation
            use std::alloc::{alloc, Layout};

            let layout = Layout::from_size_align(size, 4096)?;
            let ptr = unsafe { alloc(layout) };

            if ptr.is_null() {
                Err(anyhow::anyhow!("Memory allocation failed"))
            } else {
                Ok(ptr)
            }
        }
    }

    #[cfg(target_os = "linux")]
    async fn allocate_linux_numa(&self, size: usize, node_id: usize) -> Result<*mut u8> {
        use std::alloc::{alloc, Layout};

        // For now, use regular allocation
        // In production, this would use numa_alloc_onnode() or similar
        let layout = Layout::from_size_align(size, 4096)?;
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            Err(anyhow::anyhow!(
                "NUMA allocation failed for node {}",
                node_id
            ))
        } else {
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                if node_id == self.get_optimal_node().unwrap_or(0) {
                    metrics.local_allocations += 1;
                } else {
                    metrics.remote_allocations += 1;
                }
            }

            Ok(ptr)
        }
    }

    /// Set CPU affinity for current thread to specific NUMA node
    pub async fn set_cpu_affinity(&self, node_id: usize) -> Result<()> {
        if let Some(node) = self.topology.get_node(node_id) {
            #[cfg(target_os = "linux")]
            {
                use nix::sched::{sched_setaffinity, CpuSet};
                use nix::unistd::Pid;

                let mut cpu_set = CpuSet::new();
                for &cpu_id in &node.cpu_cores {
                    cpu_set.set(cpu_id)?;
                }

                sched_setaffinity(Pid::from_raw(0), &cpu_set)
                    .map_err(|e| anyhow::anyhow!("Failed to set CPU affinity: {}", e))?;

                info!(
                    "Set CPU affinity to NUMA node {} (CPUs: {:?})",
                    node_id, node.cpu_cores
                );
            }

            #[cfg(not(target_os = "linux"))]
            {
                debug!("CPU affinity setting not supported on this platform");
            }

            Ok(())
        } else {
            Err(anyhow::anyhow!("NUMA node {} not found", node_id))
        }
    }

    /// Optimize CPU affinity for all threads
    pub async fn optimize_cpu_affinity(&self) -> Result<()> {
        info!("Optimizing CPU affinity for NUMA topology");

        // Set current thread to optimal NUMA node
        if let Some(node_id) = self.get_optimal_node() {
            self.set_cpu_affinity(node_id).await?;
        }

        Ok(())
    }

    /// Get NUMA performance metrics
    pub async fn get_metrics(&self) -> Result<NumaMetrics> {
        let mut metrics = self.metrics.read().await.clone();

        // Update system-level metrics
        metrics.total_memory = self.topology.total_memory_mb * 1024 * 1024;

        // Calculate available memory across all nodes
        metrics.available_memory = self
            .topology
            .nodes()
            .values()
            .map(|node| node.memory_free_mb * 1024 * 1024)
            .sum();

        Ok(metrics)
    }

    /// Get memory usage for specific NUMA node
    pub async fn get_node_memory_usage(&self, node_id: usize) -> Result<(u64, u64)> {
        if let Some(node) = self.topology.get_node(node_id) {
            Ok((
                node.memory_size_mb * 1024 * 1024,
                node.memory_free_mb * 1024 * 1024,
            ))
        } else {
            Err(anyhow::anyhow!("NUMA node {} not found", node_id))
        }
    }

    /// Check if memory allocation is NUMA-local
    pub fn is_allocation_local(&self, ptr: *const u8, node_id: usize) -> bool {
        // Simplified check - in production would use numa_get_node_of_addr
        self.get_optimal_node()
            .map_or(false, |optimal| optimal == node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_numa_topology_detection() {
        let topology = NumaTopology::detect().await.unwrap();

        assert!(topology.node_count() > 0);
        assert!(topology.total_cpus > 0);
        assert!(topology.total_memory_mb > 0);

        // Test node access
        if let Some(node) = topology.get_node(0) {
            assert_eq!(node.node_id, 0);
            assert!(!node.cpu_cores.is_empty());
        }
    }

    #[tokio::test]
    async fn test_numa_manager_creation() {
        let manager = NumaManager::new(0).await.unwrap();

        assert!(manager.node_count() > 0);
        assert!(manager.get_optimal_node().is_some());

        let metrics = manager.get_metrics().await.unwrap();
        assert!(metrics.total_memory > 0);
    }

    #[tokio::test]
    async fn test_numa_memory_allocation() {
        let manager = NumaManager::new(0).await.unwrap();

        if let Some(node_id) = manager.get_optimal_node() {
            let ptr = manager.allocate_on_node(4096, node_id).await.unwrap();
            assert!(!ptr.is_null());

            // Clean up
            unsafe {
                use std::alloc::{dealloc, Layout};
                let layout = Layout::from_size_align(4096, 4096).unwrap();
                dealloc(ptr, layout);
            }
        }
    }
}
