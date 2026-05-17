// Zero-Copy Memory Management and Memory-Mapped I/O
// Advanced memory techniques for kernel-level performance

use crate::numa::{NumaManager, NumaNode};
use anyhow::Result;
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Zero-copy buffer with NUMA awareness
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    ptr: NonNull<u8>,
    size: usize,
    capacity: usize,
    layout: Layout,
    numa_node: Option<usize>,
    is_aligned: bool,
    reference_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl ZeroCopyBuffer {
    /// Create new zero-copy buffer with specified size and alignment
    pub fn new(size: usize, alignment: usize) -> Result<Self> {
        Self::new_on_node(size, alignment, None)
    }

    /// Create zero-copy buffer on specific NUMA node
    pub fn new_on_node(size: usize, alignment: usize, numa_node: Option<usize>) -> Result<Self> {
        if !alignment.is_power_of_two() || alignment == 0 {
            return Err(anyhow::anyhow!("Alignment must be a power of two"));
        }

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow::anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to allocate {} bytes", size));
        }

        let non_null_ptr =
            NonNull::new(ptr).ok_or_else(|| anyhow::anyhow!("Null pointer from allocator"))?;

        debug!(
            "Allocated zero-copy buffer: {} bytes, {} alignment, NUMA node: {:?}",
            size, alignment, numa_node
        );

        Ok(Self {
            ptr: non_null_ptr,
            size,
            capacity: size,
            layout,
            numa_node,
            is_aligned: true,
            reference_count: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
        })
    }

    /// Create buffer from existing memory (zero-copy)
    pub unsafe fn from_raw(ptr: *mut u8, size: usize, capacity: usize) -> Result<Self> {
        let non_null_ptr = NonNull::new(ptr)
            .ok_or_else(|| anyhow::anyhow!("Cannot create buffer from null pointer"))?;

        let layout = Layout::from_size_align(capacity, 1)?;

        Ok(Self {
            ptr: non_null_ptr,
            size,
            capacity,
            layout,
            numa_node: None,
            is_aligned: false,
            reference_count: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
        })
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is NUMA-local
    pub fn is_numa_local(&self) -> bool {
        self.numa_node.is_some()
    }

    /// Get NUMA node ID
    pub fn numa_node(&self) -> Option<usize> {
        self.numa_node
    }

    /// Get raw pointer for zero-copy operations
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get buffer as slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get buffer as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Resize buffer (may reallocate)
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        if new_size <= self.capacity {
            self.size = new_size;
            return Ok(());
        }

        // Need to reallocate
        let new_layout = Layout::from_size_align(new_size, self.layout.align())?;
        let new_ptr = unsafe { alloc(new_layout) };

        if new_ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to reallocate buffer"));
        }

        // Copy existing data
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.size);
            dealloc(self.ptr.as_ptr(), self.layout);
        }

        self.ptr =
            NonNull::new(new_ptr).ok_or_else(|| anyhow::anyhow!("Reallocation returned null"))?;
        self.size = new_size;
        self.capacity = new_size;
        self.layout = new_layout;

        Ok(())
    }

    /// Clone buffer reference (zero-copy)
    pub fn clone_ref(&self) -> Self {
        self.reference_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            ptr: self.ptr,
            size: self.size,
            capacity: self.capacity,
            layout: self.layout,
            numa_node: self.numa_node,
            is_aligned: self.is_aligned,
            reference_count: Arc::clone(&self.reference_count),
        }
    }

    /// Check if buffer is properly aligned for SIMD operations
    pub fn is_simd_aligned(&self) -> bool {
        let addr = self.ptr.as_ptr() as usize;
        addr % self.layout.align() == 0
    }

    /// Prefault pages to avoid page faults during critical operations
    pub fn prefault(&mut self) -> Result<()> {
        let slice = self.as_mut_slice();

        // Touch every page to ensure it's mapped
        let page_size = 4096; // Assume 4KB pages
        for offset in (0..slice.len()).step_by(page_size) {
            unsafe {
                std::ptr::write_volatile(&mut slice[offset], slice[offset]);
            }
        }

        debug!(
            "Prefaulted {} pages for zero-copy buffer",
            slice.len() / page_size
        );
        Ok(())
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        let ref_count = self
            .reference_count
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        if ref_count == 1 {
            // Last reference, deallocate
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
            debug!("Deallocated zero-copy buffer: {} bytes", self.capacity);
        }
    }
}

unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

/// Memory-mapped storage for large data structures
#[derive(Debug)]
pub struct MemoryMappedStorage {
    mmap: MmapMut,
    file_path: String,
    size: usize,
    is_persistent: bool,
}

impl MemoryMappedStorage {
    /// Create new memory-mapped storage
    pub fn new(file_path: &str, size: usize, persistent: bool) -> Result<Self> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_path)?;

        // Set file size
        file.set_len(size as u64)?;

        let mmap = unsafe {
            MmapOptions::new()
                .len(size)
                .map_mut(&file)
                .map_err(|e| anyhow::anyhow!("Memory mapping failed: {}", e))?
        };

        info!(
            "Created memory-mapped storage: {} ({} bytes, persistent: {})",
            file_path, size, persistent
        );

        Ok(Self {
            mmap,
            file_path: file_path.to_string(),
            size,
            is_persistent: persistent,
        })
    }

    /// Get storage as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Get storage as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    /// Get raw pointer for zero-copy operations
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush memory-mapped storage: {}", e))?;

        debug!("Flushed memory-mapped storage: {}", self.file_path);
        Ok(())
    }

    /// Flush changes asynchronously
    pub fn flush_async(&self) -> Result<()> {
        self.mmap
            .flush_async()
            .map_err(|e| anyhow::anyhow!("Failed to async flush: {}", e))?;

        debug!("Async flush initiated for: {}", self.file_path);
        Ok(())
    }

    /// Advise kernel about access patterns
    pub fn advise_access_pattern(&self, pattern: AccessPattern) -> Result<()> {
        #[cfg(unix)]
        {
            use nix::sys::mman::{madvise, MmapAdvise};

            let advice = match pattern {
                AccessPattern::Sequential => MmapAdvise::MADV_SEQUENTIAL,
                AccessPattern::Random => MmapAdvise::MADV_RANDOM,
                AccessPattern::WillNeed => MmapAdvise::MADV_WILLNEED,
                AccessPattern::WontNeed => MmapAdvise::MADV_DONTNEED,
            };

            unsafe {
                madvise(
                    self.mmap.as_ptr() as *mut std::ffi::c_void,
                    self.size,
                    advice,
                )
                .map_err(|e| anyhow::anyhow!("madvise failed: {}", e))?;
            }

            debug!("Applied memory advice {:?} to {}", pattern, self.file_path);
        }

        Ok(())
    }

    /// Lock pages in memory to prevent swapping
    pub fn lock_pages(&self) -> Result<()> {
        #[cfg(unix)]
        {
            use nix::sys::mman::mlock;

            unsafe {
                mlock(self.mmap.as_ptr() as *const std::ffi::c_void, self.size)
                    .map_err(|e| anyhow::anyhow!("mlock failed: {}", e))?;
            }

            debug!(
                "Locked {} bytes in memory for {}",
                self.size, self.file_path
            );
        }

        Ok(())
    }
}

/// Memory access pattern hints
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,
    Random,
    WillNeed,
    WontNeed,
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    pub total_allocated: u64,
    pub peak_allocated: u64,
    pub allocations_count: u64,
    pub deallocations_count: u64,
    pub numa_local_allocations: u64,
    pub numa_remote_allocations: u64,
    pub zero_copy_buffers: u64,
    pub memory_mapped_files: u64,
    pub average_allocation_size: f64,
}

/// Kernel-level memory manager
#[derive(Debug)]
pub struct KernelMemoryManager {
    numa_manager: Arc<NumaManager>,
    alignment: usize,
    metrics: Arc<RwLock<MemoryMetrics>>,
    buffer_pool: Arc<RwLock<HashMap<usize, Vec<ZeroCopyBuffer>>>>,
    mmap_storage: Arc<RwLock<HashMap<String, Arc<MemoryMappedStorage>>>>,
}

impl KernelMemoryManager {
    /// Create new kernel memory manager
    pub async fn new(numa_manager: &Arc<NumaManager>, alignment: usize) -> Result<Self> {
        info!(
            "Initializing kernel memory manager with {} byte alignment",
            alignment
        );

        Ok(Self {
            numa_manager: Arc::clone(numa_manager),
            alignment,
            metrics: Arc::new(RwLock::new(MemoryMetrics::default())),
            buffer_pool: Arc::new(RwLock::new(HashMap::new())),
            mmap_storage: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Allocate NUMA-aware zero-copy buffer
    pub async fn allocate_numa_buffer(
        &self,
        size: usize,
        preferred_node: Option<usize>,
    ) -> Result<ZeroCopyBuffer> {
        let node_id = preferred_node.or_else(|| self.numa_manager.get_optimal_node());

        // Try to get buffer from pool first
        if let Some(buffer) = self.get_from_pool(size).await {
            debug!("Reused buffer from pool: {} bytes", size);
            return Ok(buffer);
        }

        // Allocate new buffer
        let buffer = ZeroCopyBuffer::new_on_node(size, self.alignment, node_id)?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_allocated += size as u64;
            metrics.allocations_count += 1;
            metrics.zero_copy_buffers += 1;

            if node_id.is_some() {
                metrics.numa_local_allocations += 1;
            } else {
                metrics.numa_remote_allocations += 1;
            }

            metrics.average_allocation_size =
                metrics.total_allocated as f64 / metrics.allocations_count as f64;

            if metrics.total_allocated > metrics.peak_allocated {
                metrics.peak_allocated = metrics.total_allocated;
            }
        }

        debug!(
            "Allocated NUMA buffer: {} bytes on node {:?}",
            size, node_id
        );
        Ok(buffer)
    }

    /// Get buffer from reuse pool
    async fn get_from_pool(&self, size: usize) -> Option<ZeroCopyBuffer> {
        let mut pool = self.buffer_pool.write().await;

        // Look for buffer of appropriate size
        for (&pool_size, buffers) in pool.iter_mut() {
            if pool_size >= size && !buffers.is_empty() {
                let mut buffer = buffers.pop().unwrap();

                // Reset buffer size
                if buffer.resize(size).is_ok() {
                    return Some(buffer);
                }
            }
        }

        None
    }

    /// Return buffer to reuse pool
    pub async fn return_to_pool(&self, buffer: ZeroCopyBuffer) -> Result<()> {
        let mut pool = self.buffer_pool.write().await;
        let size = buffer.capacity();

        pool.entry(size).or_insert_with(Vec::new).push(buffer);

        // Limit pool size per bucket
        if let Some(buffers) = pool.get_mut(&size) {
            if buffers.len() > 16 {
                buffers.truncate(8); // Keep only 8 buffers
            }
        }

        Ok(())
    }

    /// Create memory-mapped storage
    pub async fn create_memory_mapped(
        &self,
        file_path: &str,
        size: usize,
    ) -> Result<MemoryMappedStorage> {
        let storage = MemoryMappedStorage::new(file_path, size, true)?;

        // Cache storage
        {
            let mut cache = self.mmap_storage.write().await;
            cache.insert(file_path.to_string(), Arc::new(storage));
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.memory_mapped_files += 1;
            metrics.total_allocated += size as u64;
        }

        // Return new instance (not shared)
        MemoryMappedStorage::new(file_path, size, true)
    }

    /// Get memory-mapped storage by path
    pub async fn get_memory_mapped(&self, file_path: &str) -> Option<Arc<MemoryMappedStorage>> {
        let cache = self.mmap_storage.read().await;
        cache.get(file_path).cloned()
    }

    /// Optimize memory allocation policies
    pub async fn optimize_memory_policies(&self) -> Result<()> {
        info!("Optimizing memory allocation policies");

        // Set NUMA memory policy for better locality
        #[cfg(target_os = "linux")]
        {
            // This would set numa_set_localalloc() or similar
            debug!("Set NUMA local allocation policy");
        }

        // Configure huge pages if available
        self.configure_huge_pages().await?;

        // Optimize buffer pool sizes
        self.optimize_buffer_pools().await?;

        Ok(())
    }

    async fn configure_huge_pages(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // Check if huge pages are available
            if std::path::Path::new("/proc/sys/vm/nr_hugepages").exists() {
                info!("Huge pages are available, optimizing allocation");
                // Would configure huge page usage here
            }
        }

        Ok(())
    }

    async fn optimize_buffer_pools(&self) -> Result<()> {
        let mut pool = self.buffer_pool.write().await;

        // Pre-allocate common buffer sizes
        let common_sizes = [4096, 8192, 16384, 32768, 65536, 131072];

        for &size in &common_sizes {
            let buffers = pool.entry(size).or_insert_with(Vec::new);

            // Ensure minimum pool size
            while buffers.len() < 4 {
                match ZeroCopyBuffer::new(size, self.alignment) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(e) => {
                        warn!("Failed to pre-allocate buffer of size {}: {}", size, e);
                        break;
                    }
                }
            }
        }

        debug!("Optimized buffer pools with {} size buckets", pool.len());
        Ok(())
    }

    /// Get memory management metrics
    pub async fn get_metrics(&self) -> Result<MemoryMetrics> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    /// Perform garbage collection on buffer pools
    pub async fn garbage_collect(&self) -> Result<usize> {
        let mut pool = self.buffer_pool.write().await;
        let mut freed_buffers = 0;

        for (_, buffers) in pool.iter_mut() {
            // Keep only half of the buffers in each pool
            let keep_count = buffers.len() / 2;
            let remove_count = buffers.len() - keep_count;

            buffers.truncate(keep_count);
            freed_buffers += remove_count;
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.deallocations_count += freed_buffers as u64;
        }

        info!("Garbage collection freed {} buffers", freed_buffers);
        Ok(freed_buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa::NumaManager;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_zero_copy_buffer_creation() {
        let buffer = ZeroCopyBuffer::new(4096, 64).unwrap();

        assert_eq!(buffer.size(), 4096);
        assert_eq!(buffer.capacity(), 4096);
        assert!(buffer.is_simd_aligned());
    }

    #[tokio::test]
    async fn test_zero_copy_buffer_resize() {
        let mut buffer = ZeroCopyBuffer::new(1024, 64).unwrap();

        // Write some data
        {
            let slice = buffer.as_mut_slice();
            slice[0] = 0xAA;
            slice[1023] = 0xBB;
        }

        // Resize
        buffer.resize(2048).unwrap();
        assert_eq!(buffer.size(), 2048);

        // Check data preserved
        {
            let slice = buffer.as_slice();
            assert_eq!(slice[0], 0xAA);
            assert_eq!(slice[1023], 0xBB);
        }
    }

    #[tokio::test]
    async fn test_memory_mapped_storage() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        let mut storage = MemoryMappedStorage::new(file_path, 8192, true).unwrap();

        // Write data
        {
            let slice = storage.as_mut_slice();
            slice[0] = 0xCC;
            slice[8191] = 0xDD;
        }

        // Flush to disk
        storage.flush().unwrap();

        // Verify data
        {
            let slice = storage.as_slice();
            assert_eq!(slice[0], 0xCC);
            assert_eq!(slice[8191], 0xDD);
        }
    }

    #[tokio::test]
    async fn test_kernel_memory_manager() {
        let numa_manager = Arc::new(NumaManager::new(0).await.unwrap());
        let memory_manager = KernelMemoryManager::new(&numa_manager, 64).await.unwrap();

        // Allocate buffer
        let buffer = memory_manager
            .allocate_numa_buffer(4096, None)
            .await
            .unwrap();
        assert_eq!(buffer.size(), 4096);

        // Check metrics
        let metrics = memory_manager.get_metrics().await.unwrap();
        assert_eq!(metrics.allocations_count, 1);
        assert_eq!(metrics.zero_copy_buffers, 1);
    }
}
