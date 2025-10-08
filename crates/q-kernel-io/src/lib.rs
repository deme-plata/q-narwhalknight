// Q-NarwhalKnight Kernel-Level I/O Optimizations
// Phase 4: Zero-copy networking and NUMA-aware memory management

//! # Q-Kernel-IO: Kernel-Level Performance Optimizations
//!
//! This crate provides kernel-level optimizations for the Q-NarwhalKnight consensus system,
//! targeting 500,000+ TPS through advanced I/O techniques and memory management.
//!
//! ## Core Optimizations
//!
//! - **io_uring Zero-Copy Networking**: Linux's async I/O interface for zero-copy operations
//! - **NUMA-Aware Memory Allocation**: CPU topology-aware memory placement
//! - **Memory-Mapped I/O**: Direct memory mapping for large data structures
//! - **Kernel Bypass Techniques**: User-space networking and direct hardware access
//! - **Zero-Copy Data Paths**: Eliminate memory copying in critical paths
//!
//! ## Performance Targets
//!
//! - **Network Throughput**: 100+ GB/s with zero-copy networking
//! - **Memory Latency**: <50ns NUMA-local access
//! - **CPU Efficiency**: 95%+ utilization across NUMA domains
//! - **I/O Latency**: <10μs storage I/O with io_uring
//! - **Consensus TPS**: 500,000+ transactions per second

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

#[cfg(feature = "benchmarks")]
pub mod benchmarks;
pub mod memory;
pub mod networking;
pub mod numa;
pub mod uring;

// Re-export key components
pub use memory::{KernelMemoryManager, MemoryMappedStorage, ZeroCopyBuffer};
pub use networking::{KernelBypassSocket, ZeroCopyNetworking};
pub use numa::{NumaManager, NumaNode, NumaTopology};
pub use uring::{IoUringEngine, UringConfig, UringOperation};

/// Kernel I/O engine configuration
#[derive(Debug, Clone)]
pub struct KernelIoConfig {
    /// Enable io_uring for async I/O operations
    pub enable_io_uring: bool,
    /// Enable NUMA-aware memory allocation
    pub enable_numa_aware: bool,
    /// Enable zero-copy networking optimizations
    pub enable_zero_copy: bool,
    /// Enable memory-mapped I/O for large files
    pub enable_memory_mapped: bool,
    /// Enable kernel bypass techniques
    pub enable_kernel_bypass: bool,
    /// io_uring queue depth (number of concurrent operations)
    pub uring_queue_depth: u32,
    /// Number of NUMA nodes to utilize
    pub numa_nodes: usize,
    /// Memory alignment for zero-copy operations
    pub zero_copy_alignment: usize,
}

impl Default for KernelIoConfig {
    fn default() -> Self {
        Self {
            enable_io_uring: cfg!(target_os = "linux"),
            enable_numa_aware: true,
            enable_zero_copy: true,
            enable_memory_mapped: true,
            enable_kernel_bypass: false, // Requires elevated privileges
            uring_queue_depth: 4096,     // High concurrency
            numa_nodes: 0,               // Auto-detect
            zero_copy_alignment: 4096,   // Page-aligned
        }
    }
}

/// Main kernel I/O optimization engine
#[derive(Debug)]
pub struct KernelIoEngine {
    config: KernelIoConfig,
    numa_manager: Arc<NumaManager>,
    memory_manager: Arc<KernelMemoryManager>,
    io_uring: Option<Arc<IoUringEngine>>,
    zero_copy_net: Option<Arc<ZeroCopyNetworking>>,
}

impl KernelIoEngine {
    /// Create new kernel I/O engine with system detection
    pub async fn new(config: KernelIoConfig) -> Result<Self> {
        info!("Initializing kernel I/O optimization engine");

        // Detect system capabilities
        let system_info = Self::detect_system_capabilities().await?;
        info!("System capabilities: {:#?}", system_info);

        // Initialize NUMA manager
        let numa_manager = Arc::new(NumaManager::new(config.numa_nodes).await?);

        // Initialize memory manager with NUMA awareness
        let memory_manager =
            Arc::new(KernelMemoryManager::new(&numa_manager, config.zero_copy_alignment).await?);

        // Initialize io_uring if enabled and supported
        let io_uring = if config.enable_io_uring && system_info.has_io_uring {
            Some(Arc::new(
                IoUringEngine::new(config.uring_queue_depth).await?,
            ))
        } else {
            if config.enable_io_uring && !system_info.has_io_uring {
                warn!("io_uring requested but not supported on this system");
            }
            None
        };

        // Initialize zero-copy networking if enabled
        let zero_copy_net = if config.enable_zero_copy {
            Some(Arc::new(ZeroCopyNetworking::new(&memory_manager).await?))
        } else {
            None
        };

        info!("Kernel I/O engine initialized successfully");
        info!("  NUMA nodes: {}", numa_manager.node_count());
        info!("  io_uring: {}", io_uring.is_some());
        info!("  Zero-copy networking: {}", zero_copy_net.is_some());

        Ok(Self {
            config,
            numa_manager,
            memory_manager,
            io_uring,
            zero_copy_net,
        })
    }

    /// Get reference to NUMA manager
    pub fn numa_manager(&self) -> &NumaManager {
        &self.numa_manager
    }

    /// Get reference to memory manager
    pub fn memory_manager(&self) -> &KernelMemoryManager {
        &self.memory_manager
    }

    /// Get reference to io_uring engine
    pub fn io_uring(&self) -> Option<&IoUringEngine> {
        self.io_uring.as_ref().map(|arc| arc.as_ref())
    }

    /// Get reference to zero-copy networking
    pub fn zero_copy_networking(&self) -> Option<&ZeroCopyNetworking> {
        self.zero_copy_net.as_ref().map(|arc| arc.as_ref())
    }

    /// Allocate NUMA-local memory for optimal performance
    pub async fn allocate_numa_memory(
        &self,
        size: usize,
        node: Option<usize>,
    ) -> Result<ZeroCopyBuffer> {
        self.memory_manager.allocate_numa_buffer(size, node).await
    }

    /// Create memory-mapped storage for large data structures
    pub async fn create_memory_mapped_storage(
        &self,
        file_path: &str,
        size: usize,
    ) -> Result<MemoryMappedStorage> {
        self.memory_manager
            .create_memory_mapped(file_path, size)
            .await
    }

    /// Perform zero-copy network send operation
    pub async fn zero_copy_send(
        &self,
        socket: &KernelBypassSocket,
        buffer: &ZeroCopyBuffer,
    ) -> Result<usize> {
        match &self.zero_copy_net {
            Some(net) => net.send_zero_copy(socket, buffer).await,
            None => Err(anyhow::anyhow!("Zero-copy networking not enabled")),
        }
    }

    /// Perform zero-copy network receive operation
    pub async fn zero_copy_receive(
        &self,
        socket: &KernelBypassSocket,
        buffer: &mut ZeroCopyBuffer,
    ) -> Result<usize> {
        match &self.zero_copy_net {
            Some(net) => net.receive_zero_copy(socket, buffer).await,
            None => Err(anyhow::anyhow!("Zero-copy networking not enabled")),
        }
    }

    /// Perform async I/O operation using io_uring
    pub async fn async_io_operation(&self, operation: UringOperation) -> Result<u64> {
        match &self.io_uring {
            Some(uring) => uring.submit_operation(operation).await,
            None => Err(anyhow::anyhow!("io_uring not available")),
        }
    }

    /// Get comprehensive performance metrics
    pub async fn performance_metrics(&self) -> Result<KernelIoMetrics> {
        Ok(KernelIoMetrics {
            numa_metrics: self.numa_manager.get_metrics().await?,
            memory_metrics: self.memory_manager.get_metrics().await?,
            uring_metrics: if let Some(uring) = &self.io_uring {
                Some(uring.get_metrics().await?)
            } else {
                None
            },
            network_metrics: if let Some(net) = &self.zero_copy_net {
                Some(net.get_metrics().await?)
            } else {
                None
            },
        })
    }

    /// Detect system capabilities for kernel optimizations
    async fn detect_system_capabilities() -> Result<SystemCapabilities> {
        let mut capabilities = SystemCapabilities {
            has_io_uring: false,
            has_numa: false,
            has_huge_pages: false,
            has_dpdk: false,
            numa_nodes: 0,
            cpu_cores: 0,
            cache_line_size: 64,
        };

        // Detect CPU information
        let info = sysinfo::System::new_all();
        capabilities.cpu_cores = info.cpus().len();

        // Detect NUMA topology
        if let Ok(numa_topology) = NumaTopology::detect().await {
            capabilities.has_numa = numa_topology.node_count() > 1;
            capabilities.numa_nodes = numa_topology.node_count();
        }

        // Platform-specific capability detection
        #[cfg(target_os = "linux")]
        {
            capabilities.has_io_uring = Self::detect_io_uring_support().await;
            capabilities.has_huge_pages = Self::detect_huge_pages_support().await;
            capabilities.has_dpdk = Self::detect_dpdk_support().await;
        }

        Ok(capabilities)
    }

    #[cfg(target_os = "linux")]
    async fn detect_io_uring_support() -> bool {
        // Try to create a simple io_uring instance
        let builder = tokio_uring::builder();
        match tokio_uring::Runtime::new(&builder) {
            Ok(_) => {
                debug!("io_uring support detected");
                true
            }
            Err(_) => {
                debug!("io_uring not supported");
                false
            }
        }
    }

    #[cfg(target_os = "linux")]
    async fn detect_huge_pages_support() -> bool {
        std::fs::read_to_string("/proc/meminfo")
            .map(|content| content.contains("HugePages_Total"))
            .unwrap_or(false)
    }

    #[cfg(target_os = "linux")]
    async fn detect_dpdk_support() -> bool {
        // Check for DPDK libraries or UIO modules
        std::path::Path::new("/sys/module/uio").exists()
            || std::path::Path::new("/dev/uio0").exists()
    }

    /// Optimize system for consensus performance
    pub async fn optimize_system(&self) -> Result<()> {
        info!("Optimizing system for kernel-level performance");

        // Set CPU affinity for NUMA locality
        self.numa_manager.optimize_cpu_affinity().await?;

        // Configure memory allocation policies
        self.memory_manager.optimize_memory_policies().await?;

        // Configure network stack optimizations
        if let Some(net) = &self.zero_copy_net {
            net.optimize_network_stack().await?;
        }

        // Configure io_uring for optimal performance
        if let Some(uring) = &self.io_uring {
            uring.optimize_performance().await?;
        }

        info!("System optimization complete");
        Ok(())
    }
}

/// System capability information
#[derive(Debug, Clone)]
struct SystemCapabilities {
    has_io_uring: bool,
    has_numa: bool,
    has_huge_pages: bool,
    has_dpdk: bool,
    numa_nodes: usize,
    cpu_cores: usize,
    cache_line_size: usize,
}

/// Comprehensive kernel I/O performance metrics
#[derive(Debug, Clone)]
pub struct KernelIoMetrics {
    pub numa_metrics: numa::NumaMetrics,
    pub memory_metrics: memory::MemoryMetrics,
    pub uring_metrics: Option<uring::UringMetrics>,
    pub network_metrics: Option<networking::NetworkMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kernel_io_engine_creation() {
        let config = KernelIoConfig::default();
        let engine = KernelIoEngine::new(config).await.unwrap();

        assert!(engine.numa_manager().node_count() > 0);

        // Test basic functionality
        let metrics = engine.performance_metrics().await.unwrap();
        assert!(metrics.numa_metrics.total_memory > 0);
    }

    #[tokio::test]
    async fn test_numa_memory_allocation() {
        let config = KernelIoConfig::default();
        let engine = KernelIoEngine::new(config).await.unwrap();

        // Allocate memory on first NUMA node
        let buffer = engine.allocate_numa_memory(4096, Some(0)).await.unwrap();
        assert_eq!(buffer.size(), 4096);
        assert!(buffer.is_numa_local());
    }

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_io_uring_operations() {
        let config = KernelIoConfig {
            enable_io_uring: true,
            ..Default::default()
        };

        if let Ok(engine) = KernelIoEngine::new(config).await {
            if engine.io_uring().is_some() {
                // Test basic io_uring operation
                let temp_file = tempfile::NamedTempFile::new().unwrap();
                let operation = UringOperation::Write {
                    fd: temp_file.as_raw_fd(),
                    buffer: vec![0u8; 1024],
                    offset: 0,
                };

                let result = engine.async_io_operation(operation).await;
                assert!(result.is_ok());
            }
        }
    }
}

// SAFETY: KernelIoEngine is safe to Send/Sync because:
// 1. All internal state is protected by Arc and proper synchronization
// 2. The tokio_uring::Runtime is only accessed through Arc<Mutex<>> which ensures thread-safe access
// 3. We never actually move the Runtime across threads - only references through Arc
// 4. All io_uring operations are submitted through the Mutex-protected interface
unsafe impl Send for KernelIoEngine {}
unsafe impl Sync for KernelIoEngine {}
