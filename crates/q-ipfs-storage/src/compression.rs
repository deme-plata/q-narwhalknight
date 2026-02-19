use crate::{IpfsStorageError, Result};
use bytes::Bytes;
use tracing::debug;

/// Compression algorithm to use
#[derive(Debug, Clone, Copy)]
pub enum CompressionType {
    /// Zstd compression (best ratio, slower)
    Zstd,
    /// LZ4 compression (fast, good ratio)
    Lz4,
    /// No compression
    None,
}

/// Handles compression and decompression of data
pub struct Compressor {
    compression_type: CompressionType,
    compression_level: i32,
}

impl Compressor {
    /// Create a new compressor with Zstd level 3 (default)
    pub fn new() -> Self {
        Self {
            compression_type: CompressionType::Zstd,
            compression_level: 3,
        }
    }

    /// Create compressor with specific type and level
    pub fn with_config(compression_type: CompressionType, level: i32) -> Self {
        Self {
            compression_type,
            compression_level: level,
        }
    }

    /// Compress data
    pub fn compress(&self, data: &[u8]) -> Result<Bytes> {
        match self.compression_type {
            CompressionType::Zstd => {
                let compressed = zstd::bulk::compress(data, self.compression_level)
                    .map_err(|e| {
                        IpfsStorageError::Compression(format!("Zstd compression failed: {}", e))
                    })?;
                debug!(
                    "Zstd compressed {} bytes -> {} bytes ({:.1}%)",
                    data.len(),
                    compressed.len(),
                    (compressed.len() as f64 / data.len() as f64) * 100.0
                );
                Ok(Bytes::from(compressed))
            }
            CompressionType::Lz4 => {
                let compressed = lz4::block::compress(data, Some(lz4::block::CompressionMode::DEFAULT), false)
                    .map_err(|e| {
                        IpfsStorageError::Compression(format!("LZ4 compression failed: {}", e))
                    })?;
                debug!(
                    "LZ4 compressed {} bytes -> {} bytes ({:.1}%)",
                    data.len(),
                    compressed.len(),
                    (compressed.len() as f64 / data.len() as f64) * 100.0
                );
                Ok(Bytes::from(compressed))
            }
            CompressionType::None => Ok(Bytes::copy_from_slice(data)),
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Bytes> {
        match self.compression_type {
            CompressionType::Zstd => {
                let decompressed = zstd::bulk::decompress(data, 128 * 1024 * 1024) // 128 MB max
                    .map_err(|e| {
                        IpfsStorageError::Compression(format!("Zstd decompression failed: {}", e))
                    })?;
                Ok(Bytes::from(decompressed))
            }
            CompressionType::Lz4 => {
                // 🛡️ v5.1.1: Validate prepended size to prevent DoS/OOM
                const MAX_DECOMPRESSED: u32 = 200_000_000;
                if data.len() >= 4 {
                    let prepended_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    if prepended_size > MAX_DECOMPRESSED {
                        return Err(IpfsStorageError::Compression(format!(
                            "LZ4 prepended size {} exceeds safety limit of {} bytes",
                            prepended_size, MAX_DECOMPRESSED
                        )));
                    }
                }
                let decompressed = lz4::block::decompress(data, None)
                    .map_err(|e| {
                        IpfsStorageError::Compression(format!("LZ4 decompression failed: {}", e))
                    })?;
                Ok(Bytes::from(decompressed))
            }
            CompressionType::None => Ok(Bytes::copy_from_slice(data)),
        }
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_compression() {
        let compressor = Compressor::new();
        let data = b"Hello, IPFS! ".repeat(1000);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(&*decompressed, &data[..]);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_lz4_compression() {
        let compressor = Compressor::with_config(CompressionType::Lz4, 0);
        let data = b"Test data ".repeat(100);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(&*decompressed, &data[..]);
    }
}
