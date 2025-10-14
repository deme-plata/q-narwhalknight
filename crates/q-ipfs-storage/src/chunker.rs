use crate::{IpfsStorageError, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Default chunk size: 256 KB (optimal for IPFS)
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Metadata about a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Chunk index in the file
    pub index: usize,
    /// Blake3 hash of the chunk
    pub hash: String,
    /// Size of the chunk in bytes
    pub size: usize,
    /// Original file path
    pub file_path: String,
    /// Offset in the original file
    pub offset: u64,
}

/// Manages chunking of files for IPFS storage
pub struct ChunkManager {
    /// Size of each chunk in bytes
    chunk_size: usize,
}

impl ChunkManager {
    /// Create a new chunk manager with default chunk size
    pub fn new() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    /// Create a chunk manager with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Split a file into chunks
    pub async fn chunk_file(&self, file_path: &Path) -> Result<Vec<(ChunkMetadata, Bytes)>> {
        info!(
            "Chunking file: {} with chunk size {}",
            file_path.display(),
            self.chunk_size
        );

        let file_data = tokio::fs::read(file_path).await?;
        let file_path_str = file_path.to_string_lossy().to_string();

        let mut chunks = Vec::new();
        let mut offset = 0u64;

        for (index, chunk_data) in file_data.chunks(self.chunk_size).enumerate() {
            let chunk_bytes = Bytes::copy_from_slice(chunk_data);

            // Calculate Blake3 hash for verification
            let hash = blake3::hash(chunk_data);
            let hash_hex = hex::encode(hash.as_bytes());

            let metadata = ChunkMetadata {
                index,
                hash: hash_hex,
                size: chunk_data.len(),
                file_path: file_path_str.clone(),
                offset,
            };

            chunks.push((metadata, chunk_bytes));
            offset += chunk_data.len() as u64;
        }

        debug!(
            "File {} split into {} chunks",
            file_path.display(),
            chunks.len()
        );

        Ok(chunks)
    }

    /// Reassemble chunks into a file
    pub async fn reassemble_chunks(
        &self,
        chunks: Vec<(ChunkMetadata, Bytes)>,
        output_path: &Path,
    ) -> Result<()> {
        info!(
            "Reassembling {} chunks to: {}",
            chunks.len(),
            output_path.display()
        );

        // Sort chunks by index
        let mut sorted_chunks = chunks;
        sorted_chunks.sort_by_key(|(meta, _)| meta.index);

        // Verify chunks
        for (metadata, data) in &sorted_chunks {
            let hash = blake3::hash(&data);
            let hash_hex = hex::encode(hash.as_bytes());

            if hash_hex != metadata.hash {
                return Err(IpfsStorageError::ChunkVerification {
                    expected: metadata.hash.clone(),
                    actual: hash_hex,
                });
            }
        }

        // Concatenate all chunks
        let mut file_data = Vec::new();
        for (_, data) in sorted_chunks {
            file_data.extend_from_slice(&data);
        }

        // Write to output file
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(output_path, file_data).await?;

        info!(
            "File reassembled successfully: {}",
            output_path.display()
        );

        Ok(())
    }

    /// Verify a chunk's integrity
    pub fn verify_chunk(&self, metadata: &ChunkMetadata, data: &[u8]) -> Result<bool> {
        let hash = blake3::hash(data);
        let hash_hex = hex::encode(hash.as_bytes());

        Ok(hash_hex == metadata.hash)
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_chunk_and_reassemble() {
        let manager = ChunkManager::new();

        // Create a test file with 1 MB of data
        let test_data = vec![0xAB; 1024 * 1024];
        let temp_file = NamedTempFile::new().unwrap();
        tokio::fs::write(temp_file.path(), &test_data)
            .await
            .unwrap();

        // Chunk the file
        let chunks = manager.chunk_file(temp_file.path()).await.unwrap();
        assert!(chunks.len() >= 4); // 1 MB / 256 KB = 4 chunks

        // Reassemble
        let output_file = NamedTempFile::new().unwrap();
        manager
            .reassemble_chunks(chunks, output_file.path())
            .await
            .unwrap();

        // Verify
        let reassembled_data = tokio::fs::read(output_file.path()).await.unwrap();
        assert_eq!(reassembled_data, test_data);
    }

    #[tokio::test]
    async fn test_chunk_verification() {
        let manager = ChunkManager::new();

        let test_data = b"Hello, IPFS!";
        let hash = blake3::hash(test_data);
        let hash_hex = hex::encode(hash.as_bytes());

        let metadata = ChunkMetadata {
            index: 0,
            hash: hash_hex,
            size: test_data.len(),
            file_path: "test.txt".to_string(),
            offset: 0,
        };

        assert!(manager.verify_chunk(&metadata, test_data).unwrap());

        // Test with wrong data
        let wrong_data = b"Wrong data";
        assert!(!manager.verify_chunk(&metadata, wrong_data).unwrap());
    }
}
