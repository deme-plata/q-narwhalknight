/// 🚀 Project APOLLO Phase 6: DELTA-V - Pre-Compressed Storage (FUEL TANK OPTIMIZATION)
///
/// Store blocks already compressed in RocksDB:
/// - Compression happens once at write time, not on every read
/// - P2P can serve compressed data directly (zero CPU for serving)
/// - Decompression only when block is actually needed
///
/// Aerospace analogy:
/// - FUEL TANK OPTIMIZATION: Pre-process fuel for optimal combustion
/// - Just like pre-cooled propellants are more efficient
/// - Blocks are "pre-processed" for efficient serving
///
/// Key features:
/// - LZ4 compression (3-5x faster than zstd, good ratio)
/// - Magic bytes for format detection
/// - Optional: keep both compressed and raw for hot path
/// - Automatic decompression on get()
///
/// Expected improvement: Near-zero CPU for P2P block serving

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tracing::{debug, info, warn};

/// Compression algorithm identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CompressionAlgorithm {
    /// No compression (raw data)
    None = 0,

    /// LZ4 block compression (fast, good ratio)
    Lz4 = 1,

    /// Zstd compression (slower, better ratio)
    Zstd = 2,

    /// LZ4 with dictionary (best for similar blocks)
    Lz4Dict = 3,
}

impl CompressionAlgorithm {
    /// Get magic bytes for this algorithm
    pub fn magic_bytes(&self) -> &'static [u8] {
        match self {
            CompressionAlgorithm::None => b"QRAW",
            CompressionAlgorithm::Lz4 => b"QLZ4",
            CompressionAlgorithm::Zstd => b"QZST",
            CompressionAlgorithm::Lz4Dict => b"QL4D",
        }
    }

    /// Detect algorithm from magic bytes
    pub fn from_magic(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        match &bytes[0..4] {
            b"QRAW" => Some(CompressionAlgorithm::None),
            b"QLZ4" => Some(CompressionAlgorithm::Lz4),
            b"QZST" => Some(CompressionAlgorithm::Zstd),
            b"QL4D" => Some(CompressionAlgorithm::Lz4Dict),
            // Legacy zstd magic
            [0x28, 0xB5, 0x2F, 0xFD] => Some(CompressionAlgorithm::Zstd),
            _ => None,
        }
    }
}

/// Pre-compressed block storage format
/// Header (12 bytes) + compressed data
#[derive(Clone, Debug)]
pub struct PrecompressedBlock {
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,

    /// Original (uncompressed) size
    pub original_size: u32,

    /// Compressed data
    pub data: Vec<u8>,
}

impl PrecompressedBlock {
    /// Compress raw block data
    pub fn compress(raw: &[u8], algorithm: CompressionAlgorithm) -> Result<Self> {
        let original_size = raw.len() as u32;

        let data = match algorithm {
            CompressionAlgorithm::None => raw.to_vec(),

            CompressionAlgorithm::Lz4 => {
                // v7.3.4: prepend_size=false — we store original_size in our own header.
                // Using true caused decompress (with explicit size) to read the 4-byte
                // prefix as compressed data, corrupting every block after compaction.
                lz4::block::compress(raw, None, false)
                    .context("LZ4 compression failed")?
            }

            CompressionAlgorithm::Zstd => {
                zstd::bulk::compress(raw, 1) // Level 1 for speed
                    .context("Zstd compression failed")?
            }

            CompressionAlgorithm::Lz4Dict => {
                // For now, same as LZ4 (dictionary support requires shared dict)
                lz4::block::compress(raw, None, false)
                    .context("LZ4 compression failed")?
            }
        };

        let compression_ratio = if !data.is_empty() {
            raw.len() as f64 / data.len() as f64
        } else {
            1.0
        };

        debug!(
            "📦 [PRECOMPRESS] {} bytes → {} bytes ({:.1}x ratio, {:?})",
            original_size,
            data.len(),
            compression_ratio,
            algorithm
        );

        Ok(Self {
            algorithm,
            original_size,
            data,
        })
    }

    /// Decompress to raw block data
    pub fn decompress(&self) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(self.data.clone()),

            CompressionAlgorithm::Lz4 | CompressionAlgorithm::Lz4Dict => {
                // 🛡️ v5.1.1: Cap original_size to prevent DoS/OOM from corrupted metadata
                const MAX_DECOMPRESSED: usize = 200_000_000;
                if self.original_size as usize > MAX_DECOMPRESSED {
                    anyhow::bail!("LZ4 original_size {} exceeds safety limit of {} bytes",
                                  self.original_size, MAX_DECOMPRESSED);
                }
                // v7.3.7: Two-format LZ4 decompression for backwards compatibility.
                // Pre-fix binaries (before 2026-02-18 21:05) used compress(raw, None, true)
                // which prepends a 4-byte size prefix to the LZ4 data. Post-fix uses false.
                // New format: decompress(data, Some(original_size)) — data has NO size prefix
                // Old format: decompress(data, None) — data HAS a 4-byte size prefix
                // Try new format first; fall back to old format for database compatibility.
                match lz4::block::decompress(&self.data, Some(self.original_size as i32)) {
                    Ok(decompressed) => Ok(decompressed),
                    Err(e1) => {
                        if self.data.len() >= 4 {
                            warn!("🔄 [LZ4-COMPAT] New-format decompress failed ({}), trying legacy prepend_size format", e1);
                            lz4::block::decompress(&self.data, None)
                                .map_err(|e2| anyhow::anyhow!(
                                    "LZ4 decompression failed in both formats: new={}, legacy={}", e1, e2
                                ))
                        } else {
                            Err(anyhow::anyhow!("LZ4 decompression failed: {}", e1))
                        }
                    }
                }
            }

            CompressionAlgorithm::Zstd => {
                // 🛡️ v5.1.1: Cap decompression size
                const MAX_DECOMPRESSED: usize = 200_000_000;
                let capped_size = (self.original_size as usize).min(MAX_DECOMPRESSED);
                zstd::bulk::decompress(&self.data, capped_size)
                    .context("Zstd decompression failed")
            }
        }
    }

    /// Serialize for storage (header + data)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(12 + self.data.len());

        // Magic bytes (4 bytes)
        result.extend_from_slice(self.algorithm.magic_bytes());

        // Original size (4 bytes, big-endian)
        result.extend_from_slice(&self.original_size.to_be_bytes());

        // Compressed size (4 bytes, big-endian)
        result.extend_from_slice(&(self.data.len() as u32).to_be_bytes());

        // Compressed data
        result.extend_from_slice(&self.data);

        result
    }

    /// Deserialize from stored bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 {
            bail!("PrecompressedBlock too short: {} bytes", bytes.len());
        }

        // Parse header
        let algorithm = CompressionAlgorithm::from_magic(&bytes[0..4])
            .context("Unknown compression format")?;

        let original_size = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let compressed_size = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        // Bounds check with overflow protection
        let available_data = bytes.len().saturating_sub(12);
        if compressed_size as usize > available_data {
            bail!(
                "PrecompressedBlock compressed_size {} exceeds available data {} bytes",
                compressed_size,
                available_data
            );
        }

        // Sanity check: compressed size shouldn't exceed 1GB
        const MAX_COMPRESSED_SIZE: u32 = 1024 * 1024 * 1024;
        if compressed_size > MAX_COMPRESSED_SIZE {
            bail!(
                "PrecompressedBlock compressed_size {} exceeds maximum {}",
                compressed_size,
                MAX_COMPRESSED_SIZE
            );
        }

        let data = bytes[12..12 + compressed_size as usize].to_vec();

        Ok(Self {
            algorithm,
            original_size,
            data,
        })
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.data.is_empty() {
            1.0
        } else {
            self.original_size as f64 / self.data.len() as f64
        }
    }

    /// Check if data is compressed
    pub fn is_compressed(&self) -> bool {
        self.algorithm != CompressionAlgorithm::None
    }

    /// Get raw compressed bytes for P2P serving (zero-copy path)
    pub fn raw_compressed(&self) -> &[u8] {
        &self.data
    }
}

/// Storage configuration for pre-compression
#[derive(Clone, Debug)]
pub struct PrecompressConfig {
    /// Default compression algorithm
    pub default_algorithm: CompressionAlgorithm,

    /// Minimum block size to compress (smaller blocks have poor ratios)
    pub min_compress_size: usize,

    /// Keep uncompressed copy in memory for hot path
    pub keep_uncompressed_cache: bool,

    /// Maximum uncompressed cache size (bytes)
    pub cache_size_limit: usize,
}

impl Default for PrecompressConfig {
    fn default() -> Self {
        Self {
            default_algorithm: CompressionAlgorithm::Lz4,
            min_compress_size: 256, // Don't compress tiny blocks
            keep_uncompressed_cache: false,
            cache_size_limit: 100 * 1024 * 1024, // 100 MB
        }
    }
}

/// Pre-compressed block store wrapper
pub struct PrecompressedStore {
    config: PrecompressConfig,
    /// Statistics
    stats: PrecompressStats,
}

/// Compression statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PrecompressStats {
    pub blocks_compressed: u64,
    pub blocks_decompressed: u64,
    pub bytes_before_compression: u64,
    pub bytes_after_compression: u64,
    pub compression_time_ms: u64,
    pub decompression_time_ms: u64,
}

impl PrecompressStats {
    pub fn overall_ratio(&self) -> f64 {
        if self.bytes_after_compression == 0 {
            1.0
        } else {
            self.bytes_before_compression as f64 / self.bytes_after_compression as f64
        }
    }

    pub fn avg_compression_time_us(&self) -> f64 {
        if self.blocks_compressed == 0 {
            0.0
        } else {
            (self.compression_time_ms * 1000) as f64 / self.blocks_compressed as f64
        }
    }

    pub fn avg_decompression_time_us(&self) -> f64 {
        if self.blocks_decompressed == 0 {
            0.0
        } else {
            (self.decompression_time_ms * 1000) as f64 / self.blocks_decompressed as f64
        }
    }
}

impl PrecompressedStore {
    pub fn new(config: PrecompressConfig) -> Self {
        Self {
            config,
            stats: PrecompressStats::default(),
        }
    }

    /// Compress block for storage
    pub fn compress(&mut self, raw: &[u8]) -> Result<PrecompressedBlock> {
        let start = std::time::Instant::now();

        let algorithm = if raw.len() < self.config.min_compress_size {
            CompressionAlgorithm::None
        } else {
            self.config.default_algorithm
        };

        let block = PrecompressedBlock::compress(raw, algorithm)?;

        // Update stats
        self.stats.blocks_compressed += 1;
        self.stats.bytes_before_compression += raw.len() as u64;
        self.stats.bytes_after_compression += block.data.len() as u64;
        self.stats.compression_time_ms += start.elapsed().as_millis() as u64;

        Ok(block)
    }

    /// Decompress block from storage
    pub fn decompress(&mut self, block: &PrecompressedBlock) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let raw = block.decompress()?;

        // Update stats
        self.stats.blocks_decompressed += 1;
        self.stats.decompression_time_ms += start.elapsed().as_millis() as u64;

        Ok(raw)
    }

    /// Get statistics
    pub fn get_stats(&self) -> PrecompressStats {
        self.stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = PrecompressStats::default();
    }
}

/// Detect if bytes are already pre-compressed
pub fn is_precompressed(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    }
    CompressionAlgorithm::from_magic(&bytes[0..4]).is_some()
}

/// Serve block for P2P (returns compressed bytes if available)
pub fn serve_block(stored_bytes: &[u8]) -> (&[u8], bool) {
    if is_precompressed(stored_bytes) {
        // Already compressed, serve directly (zero CPU)
        (stored_bytes, true)
    } else {
        // Raw data, client will need to handle
        (stored_bytes, false)
    }
}

/// Batch compression for multiple blocks
pub fn compress_batch(
    blocks: &[&[u8]],
    algorithm: CompressionAlgorithm,
) -> Result<Vec<PrecompressedBlock>> {
    blocks
        .iter()
        .map(|block| PrecompressedBlock::compress(block, algorithm))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_algorithms() {
        let data = b"Hello, World! This is test data for compression.".repeat(100);

        for algo in [
            CompressionAlgorithm::None,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd,
        ] {
            let compressed = PrecompressedBlock::compress(&data, algo).unwrap();
            let decompressed = compressed.decompress().unwrap();

            assert_eq!(data.as_slice(), decompressed.as_slice());

            if algo != CompressionAlgorithm::None {
                assert!(compressed.compression_ratio() > 1.0);
            }
        }
    }

    #[test]
    fn test_serialization() {
        let data = b"Test data for serialization".repeat(50);

        let original = PrecompressedBlock::compress(&data, CompressionAlgorithm::Lz4).unwrap();
        let bytes = original.to_bytes();
        let restored = PrecompressedBlock::from_bytes(&bytes).unwrap();

        assert_eq!(original.algorithm, restored.algorithm);
        assert_eq!(original.original_size, restored.original_size);
        assert_eq!(original.data, restored.data);
    }

    #[test]
    fn test_magic_detection() {
        let lz4_magic = b"QLZ4\x00\x00\x00\x10\x00\x00\x00\x08";
        assert_eq!(
            CompressionAlgorithm::from_magic(lz4_magic),
            Some(CompressionAlgorithm::Lz4)
        );

        let zstd_legacy = &[0x28, 0xB5, 0x2F, 0xFD];
        assert_eq!(
            CompressionAlgorithm::from_magic(zstd_legacy),
            Some(CompressionAlgorithm::Zstd)
        );

        let unknown = b"UNKN";
        assert_eq!(CompressionAlgorithm::from_magic(unknown), None);
    }

    #[test]
    fn test_precompressed_store() {
        let mut store = PrecompressedStore::new(PrecompressConfig::default());

        let data = b"Block data for storage test".repeat(100);
        let compressed = store.compress(&data).unwrap();
        let decompressed = store.decompress(&compressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());

        let stats = store.get_stats();
        assert_eq!(stats.blocks_compressed, 1);
        assert_eq!(stats.blocks_decompressed, 1);
        assert!(stats.overall_ratio() > 1.0);
    }

    #[test]
    fn test_is_precompressed() {
        assert!(is_precompressed(b"QLZ4\x00\x00\x00\x10"));
        assert!(is_precompressed(b"QZST\x00\x00\x00\x10"));
        assert!(!is_precompressed(b"random data"));
        assert!(!is_precompressed(b"abc"));
    }
}
