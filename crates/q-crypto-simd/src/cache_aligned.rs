// Cache-Aligned Memory Management for SIMD Operations
// Optimized memory layout for vectorized operations

use anyhow::Result;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;
use tracing::{debug, warn};
use crate::cpu_detection::CpuFeatures;

/// Cache-aligned buffer for SIMD operations
#[derive(Debug)]
pub struct CacheAlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
    alignment: usize,
    layout: Layout,
}

impl CacheAlignedBuffer {
    /// Create a new cache-aligned buffer
    pub fn new(size: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() || alignment == 0 {
            return Err(anyhow::anyhow!("Alignment must be a power of two"));
        }
        
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow::anyhow!("Invalid layout: {}", e))?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to allocate aligned memory"));
        }
        
        let non_null_ptr = NonNull::new(ptr)
            .ok_or_else(|| anyhow::anyhow!("Null pointer from allocator"))?;
        
        debug!("Allocated cache-aligned buffer: {} bytes, {} alignment", size, alignment);
        
        Ok(Self {
            ptr: non_null_ptr,
            len: 0,
            capacity: size,
            alignment,
            layout,
        })
    }
    
    /// Create buffer with CPU-specific alignment
    pub fn with_cpu_alignment(size: usize, cpu_features: &CpuFeatures) -> Result<Self> {
        let alignment = cpu_features.cache_line_size;
        Self::new(size, alignment)
    }
    
    /// Copy data into the aligned buffer
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.capacity {
            return Err(anyhow::anyhow!("Data too large for buffer"));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }
        
        self.len = data.len();
        Ok(())
    }
    
    /// Get a slice view of the buffer data
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    
    /// Get a mutable slice view of the buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }
    
    /// Get the raw pointer for SIMD operations
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    
    /// Get the mutable raw pointer for SIMD operations
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get current data length
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get memory alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Check if pointer is properly aligned for SIMD operations
    pub fn is_simd_aligned(&self) -> bool {
        let addr = self.ptr.as_ptr() as usize;
        addr % self.alignment == 0
    }
}

impl Drop for CacheAlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
        debug!("Deallocated cache-aligned buffer");
    }
}

unsafe impl Send for CacheAlignedBuffer {}
unsafe impl Sync for CacheAlignedBuffer {}

/// SIMD cache manager for optimized memory operations
#[derive(Debug)]
pub struct SimdCache {
    alignment: usize,
    buffer_pool: std::sync::Mutex<Vec<CacheAlignedBuffer>>,
    stats: std::sync::RwLock<SimdCacheStats>,
}

/// Statistics for SIMD cache operations
#[derive(Debug, Clone, Default)]
struct SimdCacheStats {
    buffers_allocated: u64,
    buffers_reused: u64,
    total_bytes_aligned: u64,
    cache_hit_ratio: f64,
}

impl SimdCache {
    /// Create new SIMD cache manager
    pub async fn new(alignment: usize) -> Result<Self> {
        debug!("Creating SIMD cache manager with {} byte alignment", alignment);
        
        Ok(Self {
            alignment,
            buffer_pool: std::sync::Mutex::new(Vec::new()),
            stats: std::sync::RwLock::new(SimdCacheStats::default()),
        })
    }
    
    /// Get an aligned buffer for the given data
    pub async fn align_buffer(&self, data: &[u8]) -> Result<CacheAlignedBuffer> {
        let required_size = Self::aligned_size(data.len(), self.alignment);
        
        // Try to reuse a buffer from the pool
        let mut buffer = {
            let mut pool = self.buffer_pool.lock().unwrap();
            
            // Look for a suitable buffer in the pool
            if let Some(index) = pool.iter().position(|buf| buf.capacity() >= required_size) {
                let buffer = pool.swap_remove(index);
                self.update_stats_reused().await;
                buffer
            } else {
                // Allocate a new buffer
                let buffer = CacheAlignedBuffer::new(required_size, self.alignment)?;
                self.update_stats_allocated().await;
                buffer
            }
        };
        
        // Copy data into the buffer
        let mut buffer = buffer;
        buffer.copy_from_slice(data)?;
        
        Ok(buffer)
    }
    
    /// Return a buffer to the pool for reuse
    pub async fn return_buffer(&self, mut buffer: CacheAlignedBuffer) -> Result<()> {
        // Clear the buffer for security
        let slice = buffer.as_mut_slice();
        slice.fill(0);
        
        // Return to pool if pool isn't too large
        let mut pool = self.buffer_pool.lock().unwrap();
        if pool.len() < 16 { // Limit pool size
            pool.push(buffer);
        }
        
        Ok(())
    }
    
    /// Align a vector of data slices for batch SIMD operations
    pub async fn align_batch(&self, data_slices: &[&[u8]]) -> Result<Vec<CacheAlignedBuffer>> {
        let mut aligned_buffers = Vec::with_capacity(data_slices.len());
        
        for &data in data_slices {
            let buffer = self.align_buffer(data).await?;
            aligned_buffers.push(buffer);
        }
        
        Ok(aligned_buffers)
    }
    
    /// Create a zero-filled aligned buffer
    pub async fn create_aligned_buffer(&self, size: usize) -> Result<CacheAlignedBuffer> {
        let aligned_size = Self::aligned_size(size, self.alignment);
        let mut buffer = CacheAlignedBuffer::new(aligned_size, self.alignment)?;
        
        // Zero-fill the buffer
        let slice = buffer.as_mut_slice();
        slice[..size].fill(0);
        
        self.update_stats_allocated().await;
        Ok(buffer)
    }
    
    /// Calculate aligned size for given data size
    fn aligned_size(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }
    
    /// Update statistics for buffer allocation
    async fn update_stats_allocated(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.buffers_allocated += 1;
        self.update_cache_hit_ratio(&mut stats);
    }
    
    /// Update statistics for buffer reuse
    async fn update_stats_reused(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.buffers_reused += 1;
        self.update_cache_hit_ratio(&mut stats);
    }
    
    /// Update cache hit ratio calculation
    fn update_cache_hit_ratio(&self, stats: &mut SimdCacheStats) {
        let total_requests = stats.buffers_allocated + stats.buffers_reused;
        if total_requests > 0 {
            stats.cache_hit_ratio = (stats.buffers_reused as f64) / (total_requests as f64);
        }
    }
    
    /// Get efficiency report for cache operations
    pub async fn efficiency_report(&self) -> Result<f64> {
        let stats = self.stats.read().unwrap();
        Ok(stats.cache_hit_ratio)
    }
    
    /// Get memory alignment used by this cache
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// Check if a pointer is properly aligned for SIMD operations
    pub fn is_aligned(&self, ptr: *const u8) -> bool {
        (ptr as usize) % self.alignment == 0
    }
    
    /// Get statistics for debugging
    pub async fn get_stats(&self) -> SimdCacheStats {
        self.stats.read().unwrap().clone()
    }
}

/// Trait for SIMD-optimized data structures
pub trait SimdAligned {
    /// Ensure data is aligned for SIMD operations
    fn ensure_simd_alignment(&mut self, alignment: usize) -> Result<()>;
    
    /// Check if data is properly aligned
    fn is_simd_aligned(&self, alignment: usize) -> bool;
}

impl SimdAligned for Vec<u8> {
    fn ensure_simd_alignment(&mut self, alignment: usize) -> Result<()> {
        let addr = self.as_ptr() as usize;
        if addr % alignment != 0 {
            // Reallocate with proper alignment
            let mut aligned = Vec::with_capacity(self.len() + alignment);
            
            // Add padding to achieve alignment
            let misalignment = addr % alignment;
            if misalignment != 0 {
                let padding = alignment - misalignment;
                aligned.resize(padding, 0);
            }
            
            aligned.extend_from_slice(self);
            *self = aligned;
        }
        Ok(())
    }
    
    fn is_simd_aligned(&self, alignment: usize) -> bool {
        (self.as_ptr() as usize) % alignment == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_detection::detect_cpu_features;
    
    #[tokio::test]
    async fn test_cache_aligned_buffer() {
        let data = b"Hello, SIMD world!";
        let mut buffer = CacheAlignedBuffer::new(64, 64).unwrap();
        
        buffer.copy_from_slice(data).unwrap();
        assert_eq!(buffer.as_slice(), data);
        assert!(buffer.is_simd_aligned());
    }
    
    #[tokio::test]
    async fn test_simd_cache() {
        let cache = SimdCache::new(64).await.unwrap();
        let data = b"Test data for SIMD cache";
        
        let buffer = cache.align_buffer(data).await.unwrap();
        assert_eq!(buffer.as_slice(), data);
        assert!(buffer.is_simd_aligned());
        
        cache.return_buffer(buffer).await.unwrap();
    }
    
    #[tokio::test]
    async fn test_batch_alignment() {
        let cache = SimdCache::new(64).await.unwrap();
        let data_slices = vec![b"data1".as_slice(), b"data2".as_slice(), b"data3".as_slice()];
        
        let aligned_buffers = cache.align_batch(&data_slices).await.unwrap();
        assert_eq!(aligned_buffers.len(), 3);
        
        for (i, buffer) in aligned_buffers.iter().enumerate() {
            assert_eq!(buffer.as_slice(), data_slices[i]);
            assert!(buffer.is_simd_aligned());
        }
    }
    
    #[tokio::test]
    async fn test_cpu_specific_alignment() {
        let cpu_features = detect_cpu_features();
        let data = b"CPU-aligned data";
        
        let buffer = CacheAlignedBuffer::with_cpu_alignment(64, &cpu_features).unwrap();
        assert_eq!(buffer.alignment(), cpu_features.cache_line_size);
    }
}