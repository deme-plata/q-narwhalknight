/// Block-Vertex Mapping for DAG-Knight Integration
///
/// This module provides bidirectional mapping between QBlocks and DAG vertices,
/// enabling proper integration of blockchain sync with DAG-Knight consensus.
///
/// Phase 1 Implementation: Population of dag_parents field
/// Phase 2 Integration: DAG-aware layered sync

use crate::{BlockHash, VertexId};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Bidirectional mapping between blocks and DAG vertices
///
/// This structure maintains the relationship between blockchain blocks
/// and their corresponding DAG vertices in the consensus layer.
///
/// Usage:
/// - During block production: Register new block-vertex pairs
/// - During sync: Query parent vertices to organize blocks by DAG structure
/// - During consensus: Translate between block and vertex representations
#[derive(Debug)]
pub struct BlockVertexMap {
    /// Block hash -> Vertex ID
    /// Used to look up which vertex corresponds to a given block
    block_to_vertex: RwLock<HashMap<BlockHash, VertexId>>,

    /// Vertex ID -> Block hash
    /// Used to look up which block corresponds to a given vertex
    vertex_to_block: RwLock<HashMap<VertexId, BlockHash>>,
}

impl BlockVertexMap {
    /// Create a new empty block-vertex mapping
    pub fn new() -> Self {
        Self {
            block_to_vertex: RwLock::new(HashMap::new()),
            vertex_to_block: RwLock::new(HashMap::new()),
        }
    }

    /// Register a block-vertex pair
    ///
    /// This should be called during block production after both the block
    /// and its corresponding vertex have been created.
    ///
    /// # Arguments
    /// * `block_hash` - The hash of the block
    /// * `vertex_id` - The ID of the corresponding DAG vertex
    ///
    /// # Example
    /// ```ignore
    /// let map = BlockVertexMap::new();
    /// map.register(block.hash(), vertex_id).await;
    /// ```
    pub async fn register(&self, block_hash: BlockHash, vertex_id: VertexId) {
        let mut btv = self.block_to_vertex.write().await;
        let mut vtb = self.vertex_to_block.write().await;
        btv.insert(block_hash, vertex_id);
        vtb.insert(vertex_id, block_hash);
    }

    /// Get vertex ID for a block
    ///
    /// Returns None if the block has no registered vertex
    /// (may happen for old blocks created before Phase 1)
    pub async fn get_vertex(&self, block_hash: &BlockHash) -> Option<VertexId> {
        self.block_to_vertex.read().await.get(block_hash).copied()
    }

    /// Get block hash for a vertex
    ///
    /// Returns None if the vertex has no registered block
    pub async fn get_block(&self, vertex_id: &VertexId) -> Option<BlockHash> {
        self.vertex_to_block.read().await.get(vertex_id).copied()
    }

    /// Get number of registered mappings
    ///
    /// Useful for monitoring and debugging
    pub async fn len(&self) -> usize {
        self.block_to_vertex.read().await.len()
    }

    /// Check if mapping is empty
    pub async fn is_empty(&self) -> bool {
        self.block_to_vertex.read().await.is_empty()
    }

    /// Clear all mappings
    ///
    /// WARNING: This should only be used for testing or database reset
    pub async fn clear(&self) {
        let mut btv = self.block_to_vertex.write().await;
        let mut vtb = self.vertex_to_block.write().await;
        btv.clear();
        vtb.clear();
    }
}

impl Default for BlockVertexMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_lookup() {
        let map = BlockVertexMap::new();

        let block_hash: BlockHash = [1u8; 32];
        let vertex_id: VertexId = [2u8; 32];

        // Register mapping
        map.register(block_hash, vertex_id).await;

        // Lookup both directions
        assert_eq!(map.get_vertex(&block_hash).await, Some(vertex_id));
        assert_eq!(map.get_block(&vertex_id).await, Some(block_hash));
    }

    #[tokio::test]
    async fn test_missing_entries() {
        let map = BlockVertexMap::new();

        let block_hash: BlockHash = [1u8; 32];
        let vertex_id: VertexId = [2u8; 32];

        // Lookup before registration
        assert_eq!(map.get_vertex(&block_hash).await, None);
        assert_eq!(map.get_block(&vertex_id).await, None);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        use std::sync::Arc;

        let map = Arc::new(BlockVertexMap::new());

        // Spawn multiple tasks registering mappings concurrently
        let mut handles = vec![];

        for i in 0..100 {
            let map_clone = Arc::clone(&map);
            handles.push(tokio::spawn(async move {
                let mut block_hash = [0u8; 32];
                let mut vertex_id = [0u8; 32];
                block_hash[0] = i as u8;
                vertex_id[0] = (i + 100) as u8;

                map_clone.register(block_hash, vertex_id).await;
            }));
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all mappings registered
        assert_eq!(map.len().await, 100);
    }

    #[tokio::test]
    async fn test_clear() {
        let map = BlockVertexMap::new();

        let block_hash: BlockHash = [1u8; 32];
        let vertex_id: VertexId = [2u8; 32];

        map.register(block_hash, vertex_id).await;
        assert_eq!(map.len().await, 1);

        map.clear().await;
        assert_eq!(map.len().await, 0);
        assert_eq!(map.get_vertex(&block_hash).await, None);
    }
}
