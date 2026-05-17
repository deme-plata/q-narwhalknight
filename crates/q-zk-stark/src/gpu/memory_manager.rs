//! GPU Memory Management for STARK Operations
//!
//! Efficient memory management for WebGPU-based STARK proving to ensure
//! optimal performance and prevent GPU memory fragmentation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU Memory Manager for STARK operations
#[derive(Clone)]
pub struct GpuMemoryManager {
    device: Arc<wgpu::Device>,
    buffer_pools: Arc<Mutex<HashMap<BufferKey, Vec<wgpu::Buffer>>>>,
    allocated_bytes: Arc<Mutex<u64>>,
    max_memory_bytes: u64,
}

/// Buffer characteristics for pooling
#[derive(Hash, Eq, PartialEq, Clone)]
struct BufferKey {
    size_bytes: u64,
    usage: wgpu::BufferUsages,
    label_hash: u64, // Hash of buffer label for identification
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub fn new(device: Arc<wgpu::Device>) -> Result<Self> {
        Ok(Self {
            device,
            buffer_pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_bytes: Arc::new(Mutex::new(0)),
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB max as per Server Beta analysis
        })
    }

    /// Allocate GPU buffer with automatic pooling
    pub fn allocate_buffer(&self, descriptor: &wgpu::BufferDescriptor) -> Result<wgpu::Buffer> {
        let key = BufferKey {
            size_bytes: descriptor.size,
            usage: descriptor.usage,
            label_hash: self.hash_label(descriptor.label.unwrap_or("unlabeled")),
        };

        // Try to reuse from pool first
        if let Some(buffer) = self.try_reuse_buffer(&key) {
            return Ok(buffer);
        }

        // Check memory limits
        {
            let allocated = *self.allocated_bytes.lock().unwrap();
            if allocated + descriptor.size > self.max_memory_bytes {
                return Err(anyhow!(
                    "GPU memory limit exceeded: {} + {} > {} bytes",
                    allocated,
                    descriptor.size,
                    self.max_memory_bytes
                ));
            }
        }

        // Allocate new buffer
        let buffer = self.device.create_buffer(descriptor);

        // Update allocated memory counter
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += descriptor.size;
        }

        Ok(buffer)
    }

    /// Release buffer back to pool for reuse
    pub fn release_buffer(&self, buffer: wgpu::Buffer, key: BufferKey) {
        let size_bytes = key.size_bytes; // Store size before moving key
        let mut pools = self.buffer_pools.lock().unwrap();
        let pool = pools.entry(key).or_insert_with(Vec::new);

        // Limit pool size to prevent excessive memory usage
        if pool.len() < 16 {
            pool.push(buffer);
        } else {
            // Buffer will be dropped and memory freed
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated = allocated.saturating_sub(size_bytes);
        }
    }

    /// Allocate staging buffer for CPU-GPU data transfer
    pub fn allocate_staging_buffer(&self, size: u64) -> Result<wgpu::Buffer> {
        self.allocate_buffer(&wgpu::BufferDescriptor {
            label: Some("STARK Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Allocate storage buffer for compute operations
    pub fn allocate_storage_buffer(&self, size: u64, read_only: bool) -> Result<wgpu::Buffer> {
        let usage = if read_only {
            wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        };

        self.allocate_buffer(&wgpu::BufferDescriptor {
            label: Some("STARK Storage Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Allocate uniform buffer for shader parameters
    pub fn allocate_uniform_buffer<T: bytemuck::Pod>(&self, data: &T) -> Result<wgpu::Buffer> {
        use wgpu::util::DeviceExt;

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("STARK Uniform Buffer"),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Update allocated memory counter
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += std::mem::size_of::<T>() as u64;
        }

        Ok(buffer)
    }

    /// Get current GPU memory usage statistics
    pub fn memory_stats(&self) -> GpuMemoryStats {
        let allocated = *self.allocated_bytes.lock().unwrap();
        let pools = self.buffer_pools.lock().unwrap();
        let pooled_buffers: usize = pools.values().map(|pool| pool.len()).sum();

        GpuMemoryStats {
            allocated_bytes: allocated,
            max_bytes: self.max_memory_bytes,
            utilization_percent: (allocated as f64 / self.max_memory_bytes as f64 * 100.0) as u32,
            pooled_buffers,
            active_pools: pools.len(),
        }
    }

    /// Optimize memory usage by cleaning up unused buffers
    pub fn optimize_memory(&self) -> Result<GpuMemoryOptimization> {
        let mut pools = self.buffer_pools.lock().unwrap();
        let mut freed_bytes = 0u64;
        let mut freed_buffers = 0usize;

        // Remove excess buffers from pools (keep only most recent 4)
        for (key, pool) in pools.iter_mut() {
            while pool.len() > 4 {
                pool.remove(0); // Remove oldest buffer
                freed_bytes += key.size_bytes;
                freed_buffers += 1;
            }
        }

        // Update allocated memory counter
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated = allocated.saturating_sub(freed_bytes);
        }

        Ok(GpuMemoryOptimization {
            freed_bytes,
            freed_buffers,
            remaining_allocated: *self.allocated_bytes.lock().unwrap(),
        })
    }

    /// Pre-allocate buffers for common STARK operations
    pub fn preallocate_stark_buffers(&self) -> Result<StarkBufferSet> {
        // Common buffer sizes for STARK operations based on Server Beta analysis
        let trace_buffer = self.allocate_storage_buffer(64 * 1024 * 1024, false)?; // 64MB execution trace
        let constraint_buffer = self.allocate_storage_buffer(32 * 1024 * 1024, true)?; // 32MB constraints
        let fft_buffer = self.allocate_storage_buffer(128 * 1024 * 1024, false)?; // 128MB FFT working space
        let commitment_buffer = self.allocate_storage_buffer(16 * 1024 * 1024, false)?; // 16MB Merkle commitments

        Ok(StarkBufferSet {
            trace_buffer,
            constraint_buffer,
            fft_buffer,
            commitment_buffer,
        })
    }

    // Private helper methods

    fn try_reuse_buffer(&self, key: &BufferKey) -> Option<wgpu::Buffer> {
        let mut pools = self.buffer_pools.lock().unwrap();
        if let Some(pool) = pools.get_mut(key) {
            pool.pop() // Reuse most recently used buffer
        } else {
            None
        }
    }

    fn hash_label(&self, label: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        label.hash(&mut hasher);
        hasher.finish()
    }
}

/// Pre-allocated buffer set for STARK operations
pub struct StarkBufferSet {
    pub trace_buffer: wgpu::Buffer,
    pub constraint_buffer: wgpu::Buffer,
    pub fft_buffer: wgpu::Buffer,
    pub commitment_buffer: wgpu::Buffer,
}

impl StarkBufferSet {
    /// Get total allocated size in bytes
    pub fn total_size_bytes(&self) -> u64 {
        // Sum of all pre-allocated buffer sizes
        64 * 1024 * 1024 + // trace_buffer
        32 * 1024 * 1024 + // constraint_buffer  
        128 * 1024 * 1024 + // fft_buffer
        16 * 1024 * 1024 // commitment_buffer
    }
}

/// GPU memory usage statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub allocated_bytes: u64,
    pub max_bytes: u64,
    pub utilization_percent: u32,
    pub pooled_buffers: usize,
    pub active_pools: usize,
}

impl GpuMemoryStats {
    /// Check if memory usage is within Phase 3 targets (<4GB as per Server Beta)
    pub fn meets_phase3_targets(&self) -> bool {
        let target_max_bytes = 4 * 1024 * 1024 * 1024; // 4GB target
        self.allocated_bytes <= target_max_bytes && self.utilization_percent <= 80
    }

    /// Get memory pressure level for optimization decisions
    pub fn memory_pressure(&self) -> MemoryPressure {
        match self.utilization_percent {
            0..=50 => MemoryPressure::Low,
            51..=75 => MemoryPressure::Medium,
            76..=90 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    /// Format stats for human-readable output
    pub fn format_stats(&self) -> String {
        format!(
            "GPU Memory: {:.1}MB / {:.1}MB ({:>3}%) | Pools: {} | Buffers: {}",
            self.allocated_bytes as f64 / (1024.0 * 1024.0),
            self.max_bytes as f64 / (1024.0 * 1024.0),
            self.utilization_percent,
            self.active_pools,
            self.pooled_buffers
        )
    }
}

/// Memory pressure levels for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPressure {
    Low,      // <50% usage - can allocate freely
    Medium,   // 50-75% usage - monitor allocation
    High,     // 75-90% usage - optimize existing buffers
    Critical, // >90% usage - aggressive cleanup needed
}

/// Result of memory optimization operation
#[derive(Debug)]
pub struct GpuMemoryOptimization {
    pub freed_bytes: u64,
    pub freed_buffers: usize,
    pub remaining_allocated: u64,
}

impl GpuMemoryOptimization {
    pub fn format_result(&self) -> String {
        format!(
            "Memory optimization: freed {:.1}MB ({} buffers), {:.1}MB remaining",
            self.freed_bytes as f64 / (1024.0 * 1024.0),
            self.freed_buffers,
            self.remaining_allocated as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Memory management configuration
pub struct GpuMemoryConfig {
    pub max_memory_bytes: u64,
    pub pool_size_limit: usize,
    pub optimization_threshold: u32, // Utilization % to trigger optimization
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            pool_size_limit: 16,
            optimization_threshold: 75, // Optimize when >75% utilized
        }
    }
}

impl GpuMemoryConfig {
    /// Phase 3 optimized configuration
    pub fn phase3_optimized() -> Self {
        Self {
            max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB target per Server Beta
            pool_size_limit: 8,                       // Smaller pools for better memory efficiency
            optimization_threshold: 60, // Earlier optimization for sustained performance
        }
    }

    /// High-performance configuration for GPU clusters
    pub fn high_performance() -> Self {
        Self {
            max_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB for large-scale operations
            pool_size_limit: 32, // Larger pools for reduced allocation overhead
            optimization_threshold: 80, // Later optimization for maximum throughput
        }
    }
}
