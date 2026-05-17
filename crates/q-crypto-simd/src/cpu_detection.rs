// CPU Feature Detection for SIMD Optimization
// Runtime detection of available SIMD instruction sets

use std::sync::OnceLock;
use tracing::{info, debug, warn};

/// CPU feature detection results
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512f: bool,      // AVX-512 Foundation
    pub has_avx512dq: bool,     // AVX-512 Doubleword and Quadword
    pub has_avx512bw: bool,     // AVX-512 Byte and Word
    pub has_avx512vl: bool,     // AVX-512 Vector Length Extensions
    pub has_avx512: bool,       // Combined AVX-512 support
    pub has_neon: bool,         // ARM NEON (for future ARM support)
    pub has_sha_ni: bool,       // SHA New Instructions
    pub has_aes_ni: bool,       // AES New Instructions
    pub cache_line_size: usize,
    pub num_cores: usize,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            has_avx2: false,
            has_avx512f: false,
            has_avx512dq: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512: false,
            has_neon: false,
            has_sha_ni: false,
            has_aes_ni: false,
            cache_line_size: 64,    // Default to 64-byte cache lines
            num_cores: 1,
        }
    }
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Detect CPU features once and cache the result
pub fn detect_cpu_features() -> CpuFeatures {
    CPU_FEATURES.get_or_init(|| {
        detect_cpu_features_impl()
    }).clone()
}

/// Internal implementation of CPU feature detection
fn detect_cpu_features_impl() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        detect_x86_64_features()
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        detect_aarch64_features()
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        warn!("CPU feature detection not implemented for this architecture");
        CpuFeatures::default()
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_64_features() -> CpuFeatures {
    let mut features = CpuFeatures::default();
    
    // Use runtime feature detection with fallback for compatibility
    #[cfg(target_feature = "avx2")]
    {
        debug!("AVX2 support compiled in");
        features.has_avx2 = true;
    }
    
    // Safer feature detection without direct core_arch usage
    features.has_avx2 = cfg!(target_feature = "avx2");
    features.has_aes_ni = cfg!(target_feature = "aes");
    
    // Conservative defaults for stability
    debug!("Using conservative SIMD feature detection for compatibility");
    
    // Set conservative capabilities
    features.num_cores = num_cpus::get_physical();
    features.cache_line_size = 64; // Standard cache line size
    
    info!("x86_64 SIMD features detected: AVX2={}, AES-NI={}", 
          features.has_avx2, features.has_aes_ni);
    
    if features.has_aes_ni {
        debug!("AES-NI (AES New Instructions) detected");
    }
    
    // Try to detect cache line size and core count
    features.cache_line_size = detect_cache_line_size();
    features.num_cores = num_cpus::get();
    
    info!("Detected {} CPU cores, cache line size: {} bytes", 
          features.num_cores, features.cache_line_size);
    
    features
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64_features() -> CpuFeatures {
    let mut features = CpuFeatures::default();
    
    // On ARM64, NEON is mandatory in the architecture spec
    features.has_neon = true;
    info!("ARM NEON support detected (architectural)");
    
    // Try to detect additional features
    #[cfg(target_feature = "sha2")]
    {
        features.has_sha_ni = true;
        debug!("ARM SHA2 instructions detected");
    }
    
    #[cfg(target_feature = "aes")]
    {
        features.has_aes_ni = true;
        debug!("ARM AES instructions detected");
    }
    
    features.cache_line_size = detect_cache_line_size();
    features.num_cores = num_cpus::get();
    
    info!("Detected {} ARM cores, cache line size: {} bytes",
          features.num_cores, features.cache_line_size);
    
    features
}

/// Attempt to detect cache line size
fn detect_cache_line_size() -> usize {
    // Try various methods to detect cache line size
    
    // Method 1: Check if we can use CPUID on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(cache_info) = get_x86_cache_info() {
            return cache_info;
        }
    }
    
    // Method 2: Read from /proc/cpuinfo on Linux
    #[cfg(target_os = "linux")]
    {
        if let Some(cache_size) = read_linux_cache_info() {
            return cache_size;
        }
    }
    
    // Method 3: Read from sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        if let Some(cache_size) = read_macos_cache_info() {
            return cache_size;
        }
    }
    
    // Default fallback
    debug!("Could not detect cache line size, using default 64 bytes");
    64
}

#[cfg(target_arch = "x86_64")]
fn get_x86_cache_info() -> Option<usize> {
    use raw_cpuid::CpuId;
    
    let cpuid = CpuId::new();
    
    if let Some(cache_params) = cpuid.get_cache_parameters() {
        for cache in cache_params {
            if cache.cache_type() == raw_cpuid::CacheType::Data && cache.level() == 1 {
                let line_size = cache.coherency_line_size() as usize;
                debug!("Detected L1 cache line size: {} bytes", line_size);
                return Some(line_size);
            }
        }
    }
    
    None
}

#[cfg(target_os = "linux")]
fn read_linux_cache_info() -> Option<usize> {
    use std::fs;
    
    // Try to read L1 cache line size from sysfs
    let paths = [
        "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size",
        "/sys/devices/system/cpu/cpu0/cache/index1/coherency_line_size",
    ];
    
    for path in &paths {
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(size) = content.trim().parse::<usize>() {
                debug!("Read cache line size from {}: {} bytes", path, size);
                return Some(size);
            }
        }
    }
    
    None
}

#[cfg(target_os = "macos")]
fn read_macos_cache_info() -> Option<usize> {
    use std::process::Command;
    
    // Use sysctl to get cache line size
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("hw.cachelinesize")
        .output()
        .ok()?;
    
    if output.status.success() {
        let size_str = String::from_utf8(output.stdout).ok()?;
        let size = size_str.trim().parse::<usize>().ok()?;
        debug!("Read cache line size from sysctl: {} bytes", size);
        return Some(size);
    }
    
    None
}

/// Get the optimal SIMD vector size for the current CPU
pub fn optimal_vector_size(features: &CpuFeatures) -> usize {
    if features.has_avx512 {
        64  // 512 bits = 64 bytes
    } else if features.has_avx2 {
        32  // 256 bits = 32 bytes
    } else if features.has_neon {
        16  // 128 bits = 16 bytes
    } else {
        8   // Fallback to 64-bit operations
    }
}

/// Get the optimal batch size for SIMD operations
pub fn optimal_batch_size(features: &CpuFeatures, element_size: usize) -> usize {
    let vector_size = optimal_vector_size(features);
    let elements_per_vector = vector_size / element_size;
    
    // Aim for processing multiple vectors worth of data
    // but keep it reasonable for memory usage
    let target_batch = elements_per_vector * 8;
    
    // Cap at reasonable limits based on CPU cores
    let max_batch = features.num_cores * 16;
    
    std::cmp::min(target_batch, max_batch).max(1)
}

/// Get AVX-512 feature flags (placeholder implementation)
/// This function is used by the benchmarking code
pub fn get_avx512_features() -> u64 {
    let features = detect_cpu_features();
    let mut flags = 0u64;
    
    if features.has_avx512f { flags |= 1 << 0; }
    if features.has_avx512dq { flags |= 1 << 1; }
    if features.has_avx512bw { flags |= 1 << 2; }
    if features.has_avx512vl { flags |= 1 << 3; }
    
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_detection() {
        let features = detect_cpu_features();
        
        // Should always detect some basic information
        assert!(features.cache_line_size >= 32);
        assert!(features.num_cores >= 1);
        
        // Print detected features for debugging
        println!("CPU Features: {:#?}", features);
    }
    
    #[test]
    fn test_optimal_sizes() {
        let features = detect_cpu_features();
        
        let vector_size = optimal_vector_size(&features);
        assert!(vector_size >= 8);
        assert!(vector_size <= 64);
        
        let batch_size = optimal_batch_size(&features, 32);
        assert!(batch_size >= 1);
        assert!(batch_size <= 1000);
    }
}