use anyhow::Result;
use q_types::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Storage backend trait for vertex persistence
#[async_trait::async_trait]
pub trait VertexStorage: Send + Sync {
    async fn store_vertex(&self, vertex: &Vertex) -> Result<()>;
    async fn get_vertex(&self, vertex_id: &VertexId) -> Result<Option<Vertex>>;
    async fn get_vertices_by_round(&self, round: Round) -> Result<Vec<Vertex>>;
    async fn get_vertices_by_author(&self, author: &NodeId) -> Result<Vec<Vertex>>;
    async fn delete_vertex(&self, vertex_id: &VertexId) -> Result<()>;
    async fn get_vertex_count(&self) -> Result<usize>;
}

/// Persistent index for fast lookups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexIndex {
    /// Vertices organized by round
    pub vertices_by_round: BTreeMap<Round, Vec<VertexId>>,
    /// Vertices organized by author
    pub vertices_by_author: HashMap<NodeId, Vec<VertexId>>,
    /// Parent-child relationships for DAG traversal
    pub children: HashMap<VertexId, Vec<VertexId>>,
    /// Fast lookup for causal dependencies
    pub causal_history: HashMap<VertexId, Vec<VertexId>>,
}

impl Default for VertexIndex {
    fn default() -> Self {
        Self {
            vertices_by_round: BTreeMap::new(),
            vertices_by_author: HashMap::new(),
            children: HashMap::new(),
            causal_history: HashMap::new(),
        }
    }
}

/// High-performance storage for DAG vertices with persistent backing
/// Maintains causal ordering and enables fast lookups
pub struct VertexStore {
    /// Persistent storage backend
    storage: Arc<dyn VertexStorage>,
    /// In-memory index for fast lookups (persisted periodically)
    index: RwLock<VertexIndex>,
    /// Flag to enable/disable in-memory caching
    cache_enabled: bool,
    /// In-memory cache for recently accessed vertices
    vertex_cache: RwLock<HashMap<VertexId, Vertex>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl VertexStore {
    /// Create a new vertex store with the given storage backend
    pub fn new(storage: Arc<dyn VertexStorage>) -> Self {
        Self {
            storage,
            index: RwLock::new(VertexIndex::default()),
            cache_enabled: true,
            vertex_cache: RwLock::new(HashMap::new()),
            max_cache_size: 1000, // v6.1.1: Reduced 10k→1k to save ~300MB on 8GB nodes
        }
    }

    /// Create a new in-memory vertex store (for testing)
    pub fn new_in_memory() -> Self {
        Self::new(Arc::new(InMemoryVertexStorage::new()))
    }

    /// Load index from persistent storage
    pub async fn load_index(&self) -> Result<()> {
        // In a real implementation, this would load the index from persistent storage
        // For now, we'll rebuild it from stored vertices
        self.rebuild_index().await
    }

    /// Rebuild index from all stored vertices
    async fn rebuild_index(&self) -> Result<()> {
        info!("🔄 Rebuilding vertex store index...");
        let mut new_index = VertexIndex::default();

        // This is a placeholder - in real implementation, we'd scan all stored vertices
        // For now, the index will be built as vertices are stored

        {
            let mut index = self.index.write().await;
            *index = new_index;
        }

        info!("✅ Vertex store index rebuilt");
        Ok(())
    }

    /// Store a vertex with all index updates
    pub async fn store_vertex(&self, vertex: Vertex) -> Result<()> {
        let vertex_id = vertex.id;

        info!(
            "💾 Storing vertex {} from round {} by author {:?}",
            hex::encode(vertex_id),
            vertex.round,
            vertex.author
        );

        // Store vertex in persistent storage
        self.storage.store_vertex(&vertex).await?;

        // Update in-memory cache if enabled
        if self.cache_enabled {
            self.update_cache(&vertex).await;
        }

        // Update indices
        self.update_indices(&vertex).await?;

        debug!(
            "✅ Successfully stored vertex {} with {} parents",
            hex::encode(vertex_id),
            vertex.parents.len()
        );

        Ok(())
    }

    /// Update in-memory cache with vertex
    async fn update_cache(&self, vertex: &Vertex) {
        let mut cache = self.vertex_cache.write().await;

        // Evict old entries if cache is full
        if cache.len() >= self.max_cache_size {
            // Simple LRU: remove oldest entries (in practice, we'd use a proper LRU)
            let keys_to_remove: Vec<VertexId> = cache
                .keys()
                .take(self.max_cache_size / 4)
                .copied()
                .collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(vertex.id, vertex.clone());
    }

    /// Update all indices with new vertex
    async fn update_indices(&self, vertex: &Vertex) -> Result<()> {
        let vertex_id = vertex.id;
        let mut index = self.index.write().await;

        // Update round index
        index
            .vertices_by_round
            .entry(vertex.round)
            .or_insert_with(Vec::new)
            .push(vertex_id);

        // Update author index
        index
            .vertices_by_author
            .entry(vertex.author)
            .or_insert_with(Vec::new)
            .push(vertex_id);

        // Update parent-child relationships
        for parent_id in &vertex.parents {
            index
                .children
                .entry(*parent_id)
                .or_insert_with(Vec::new)
                .push(vertex_id);
        }

        // Compute and store causal history
        let mut history = Vec::new();

        // Add all parents to history
        for parent_id in &vertex.parents {
            history.push(*parent_id);

            // Add all ancestors of each parent
            if let Some(parent_history) = index.causal_history.get(parent_id) {
                for ancestor in parent_history {
                    if !history.contains(ancestor) {
                        history.push(*ancestor);
                    }
                }
            }
        }

        index.causal_history.insert(vertex_id, history);

        Ok(())
    }

    /// Get vertex by ID
    pub async fn get_vertex(&self, vertex_id: &VertexId) -> Option<Vertex> {
        // Try cache first if enabled
        if self.cache_enabled {
            let cache = self.vertex_cache.read().await;
            if let Some(vertex) = cache.get(vertex_id) {
                return Some(vertex.clone());
            }
        }

        // Fallback to persistent storage
        match self.storage.get_vertex(vertex_id).await {
            Ok(Some(vertex)) => {
                // Add to cache if enabled
                if self.cache_enabled {
                    self.update_cache(&vertex).await;
                }
                Some(vertex)
            }
            Ok(None) => None,
            Err(e) => {
                warn!("Failed to get vertex {}: {}", hex::encode(vertex_id), e);
                None
            }
        }
    }

    /// Add vertex (alias for store_vertex for compatibility)
    pub async fn add_vertex(&self, vertex: Vertex) -> Result<()> {
        self.store_vertex(vertex).await
    }

    /// Get all vertices in a round
    pub async fn get_vertices_in_round(&self, round: Round) -> Vec<Vertex> {
        // First try using the index
        let index = self.index.read().await;
        if let Some(vertex_ids) = index.vertices_by_round.get(&round) {
            let mut vertices = Vec::new();
            for vertex_id in vertex_ids {
                if let Some(vertex) = self.get_vertex(vertex_id).await {
                    vertices.push(vertex);
                }
            }
            return vertices;
        }

        // Fallback to storage query if index is empty
        match self.storage.get_vertices_by_round(round).await {
            Ok(vertices) => vertices,
            Err(e) => {
                warn!("Failed to get vertices for round {}: {}", round, e);
                Vec::new()
            }
        }
    }

    /// Get vertices by author
    pub async fn get_vertices_by_author(&self, author: &NodeId) -> Vec<Vertex> {
        // First try using the index
        let index = self.index.read().await;
        if let Some(vertex_ids) = index.vertices_by_author.get(author) {
            let mut vertices = Vec::new();
            for vertex_id in vertex_ids {
                if let Some(vertex) = self.get_vertex(vertex_id).await {
                    vertices.push(vertex);
                }
            }
            return vertices;
        }

        // Fallback to storage query if index is empty
        match self.storage.get_vertices_by_author(author).await {
            Ok(vertices) => vertices,
            Err(e) => {
                warn!("Failed to get vertices for author {:?}: {}", author, e);
                Vec::new()
            }
        }
    }

    /// Get children of a vertex
    pub async fn get_children(&self, vertex_id: &VertexId) -> Vec<VertexId> {
        let index = self.index.read().await;
        index
            .children
            .get(vertex_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Check if vertex exists
    pub async fn contains_vertex(&self, vertex_id: &VertexId) -> bool {
        // Try cache first
        if self.cache_enabled {
            let cache = self.vertex_cache.read().await;
            if cache.contains_key(vertex_id) {
                return true;
            }
        }

        // Check persistent storage
        match self.storage.get_vertex(vertex_id).await {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(_) => false,
        }
    }

    /// Get causal history of a vertex (all ancestors)
    pub async fn get_causal_history(&self, vertex_id: &VertexId) -> Vec<VertexId> {
        let index = self.index.read().await;
        index
            .causal_history
            .get(vertex_id)
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Check if vertex A causally precedes vertex B
    pub async fn causally_precedes(&self, a: &VertexId, b: &VertexId) -> bool {
        let index = self.index.read().await;

        if let Some(history_b) = index.causal_history.get(b) {
            history_b.contains(a)
        } else {
            false
        }
    }

    /// Get latest round with vertices
    pub async fn get_latest_round(&self) -> Option<Round> {
        let index = self.index.read().await;
        index.vertices_by_round.keys().last().copied()
    }

    /// Get range of rounds [start, end]
    pub async fn get_round_range(&self, start: Round, end: Round) -> Vec<Vertex> {
        let mut result = Vec::new();

        for round in start..=end {
            let round_vertices = self.get_vertices_in_round(round).await;
            result.extend(round_vertices);
        }

        result
    }

    /// Get total vertex count
    pub async fn get_vertex_count(&self) -> Result<usize> {
        self.storage.get_vertex_count().await
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> VertexStoreStats {
        let index = self.index.read().await;

        let total_vertices = self.storage.get_vertex_count().await.unwrap_or(0);
        let total_rounds = index.vertices_by_round.len();
        let total_authors = index.vertices_by_author.len();

        let latest_round = index.vertices_by_round.keys().last().copied().unwrap_or(0);
        let earliest_round = index.vertices_by_round.keys().next().copied().unwrap_or(0);

        let mut vertices_per_round = BTreeMap::new();
        for (round, vertex_ids) in index.vertices_by_round.iter() {
            vertices_per_round.insert(*round, vertex_ids.len());
        }

        VertexStoreStats {
            total_vertices,
            total_rounds,
            total_authors,
            latest_round,
            earliest_round,
            vertices_per_round,
        }
    }

    /// Get current vertex cache size
    pub async fn cache_size(&self) -> usize {
        let cache = self.vertex_cache.read().await;
        cache.len()
    }

    /// Get index sizes for diagnostics: (children_entries, causal_history_entries)
    pub async fn get_index_sizes(&self) -> (usize, usize) {
        let index = self.index.read().await;
        (index.children.len(), index.causal_history.len())
    }

    /// Clean up old vertices beyond retention policy
    pub async fn cleanup_old_vertices(&self, keep_rounds: u64) -> Result<usize> {
        let latest_round = self.get_latest_round().await.unwrap_or(0);
        let cutoff_round = latest_round.saturating_sub(keep_rounds);

        let mut removed_count = 0;

        // Get vertices to remove from index
        let vertices_to_remove: Vec<VertexId> = {
            let index = self.index.read().await;
            let mut to_remove = Vec::new();

            for (round, vertex_ids) in index.vertices_by_round.iter() {
                if *round < cutoff_round {
                    to_remove.extend(vertex_ids.iter().copied());
                }
            }

            to_remove
        };

        // Remove from storage and update indices
        for vertex_id in vertices_to_remove {
            // Get vertex info before removal
            if let Some(vertex) = self.get_vertex(&vertex_id).await {
                let round = vertex.round;
                let author = vertex.author;

                // Remove from persistent storage
                self.storage.delete_vertex(&vertex_id).await?;

                // Remove from cache
                if self.cache_enabled {
                    let mut cache = self.vertex_cache.write().await;
                    cache.remove(&vertex_id);
                }

                // Update indices
                {
                    let mut index = self.index.write().await;

                    // Remove from round index
                    if let Some(round_vertices) = index.vertices_by_round.get_mut(&round) {
                        round_vertices.retain(|id| *id != vertex_id);
                        if round_vertices.is_empty() {
                            index.vertices_by_round.remove(&round);
                        }
                    }

                    // Remove from author index
                    if let Some(author_vertices) = index.vertices_by_author.get_mut(&author) {
                        author_vertices.retain(|id| *id != vertex_id);
                        if author_vertices.is_empty() {
                            index.vertices_by_author.remove(&author);
                        }
                    }

                    // Remove from children index
                    index.children.remove(&vertex_id);

                    // Remove from causal history
                    index.causal_history.remove(&vertex_id);
                }

                removed_count += 1;
            }
        }

        info!(
            "Cleaned up {} vertices older than round {}",
            removed_count, cutoff_round
        );
        Ok(removed_count)
    }
}

/// In-memory implementation of VertexStorage for testing
pub struct InMemoryVertexStorage {
    vertices: RwLock<HashMap<VertexId, Vertex>>,
}

impl InMemoryVertexStorage {
    pub fn new() -> Self {
        Self {
            vertices: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl VertexStorage for InMemoryVertexStorage {
    async fn store_vertex(&self, vertex: &Vertex) -> Result<()> {
        let mut vertices = self.vertices.write().await;
        vertices.insert(vertex.id, vertex.clone());
        Ok(())
    }

    async fn get_vertex(&self, vertex_id: &VertexId) -> Result<Option<Vertex>> {
        let vertices = self.vertices.read().await;
        Ok(vertices.get(vertex_id).cloned())
    }

    async fn get_vertices_by_round(&self, round: Round) -> Result<Vec<Vertex>> {
        let vertices = self.vertices.read().await;
        let round_vertices = vertices
            .values()
            .filter(|v| v.round == round)
            .cloned()
            .collect();
        Ok(round_vertices)
    }

    async fn get_vertices_by_author(&self, author: &NodeId) -> Result<Vec<Vertex>> {
        let vertices = self.vertices.read().await;
        let author_vertices = vertices
            .values()
            .filter(|v| &v.author == author)
            .cloned()
            .collect();
        Ok(author_vertices)
    }

    async fn delete_vertex(&self, vertex_id: &VertexId) -> Result<()> {
        let mut vertices = self.vertices.write().await;
        vertices.remove(vertex_id);
        Ok(())
    }

    async fn get_vertex_count(&self) -> Result<usize> {
        let vertices = self.vertices.read().await;
        Ok(vertices.len())
    }
}

#[derive(Debug, Clone)]
pub struct VertexStoreStats {
    pub total_vertices: usize,
    pub total_rounds: usize,
    pub total_authors: usize,
    pub latest_round: Round,
    pub earliest_round: Round,
    pub vertices_per_round: BTreeMap<Round, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vertex(id: u8, round: Round, author: u8, parents: Vec<u8>) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [author; 32],
            tx_root: [0u8; 32],
            parents: parents.into_iter().map(|p| [p; 32]).collect(),
            transactions: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_vertex_storage() {
        let store = VertexStore::new_in_memory();
        let vertex = create_test_vertex(1, 0, 1, vec![]);

        store.store_vertex(vertex.clone()).await.unwrap();

        let retrieved = store.get_vertex(&[1u8; 32]).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, vertex.id);
    }

    #[tokio::test]
    async fn test_round_queries() {
        let store = VertexStore::new_in_memory();

        // Store vertices in different rounds
        let v1 = create_test_vertex(1, 0, 1, vec![]);
        let v2 = create_test_vertex(2, 0, 2, vec![]);
        let v3 = create_test_vertex(3, 1, 1, vec![1]);

        store.store_vertex(v1).await.unwrap();
        store.store_vertex(v2).await.unwrap();
        store.store_vertex(v3).await.unwrap();

        let round_0_vertices = store.get_vertices_in_round(0).await;
        assert_eq!(round_0_vertices.len(), 2);

        let round_1_vertices = store.get_vertices_in_round(1).await;
        assert_eq!(round_1_vertices.len(), 1);

        let latest_round = store.get_latest_round().await;
        assert_eq!(latest_round, Some(1));
    }

    #[tokio::test]
    async fn test_causal_relationships() {
        let store = VertexStore::new_in_memory();

        // Create a simple chain: v1 -> v2 -> v3
        let v1 = create_test_vertex(1, 0, 1, vec![]);
        let v2 = create_test_vertex(2, 1, 2, vec![1]);
        let v3 = create_test_vertex(3, 2, 1, vec![2]);

        store.store_vertex(v1).await.unwrap();
        store.store_vertex(v2).await.unwrap();
        store.store_vertex(v3).await.unwrap();

        // Check causal relationships
        assert!(store.causally_precedes(&[1u8; 32], &[2u8; 32]).await);
        assert!(store.causally_precedes(&[1u8; 32], &[3u8; 32]).await);
        assert!(store.causally_precedes(&[2u8; 32], &[3u8; 32]).await);

        assert!(!store.causally_precedes(&[2u8; 32], &[1u8; 32]).await);
        assert!(!store.causally_precedes(&[3u8; 32], &[1u8; 32]).await);
    }

    #[tokio::test]
    async fn test_children_tracking() {
        let store = VertexStore::new_in_memory();

        let v1 = create_test_vertex(1, 0, 1, vec![]);
        let v2 = create_test_vertex(2, 1, 2, vec![1]);
        let v3 = create_test_vertex(3, 1, 3, vec![1]);

        store.store_vertex(v1).await.unwrap();
        store.store_vertex(v2).await.unwrap();
        store.store_vertex(v3).await.unwrap();

        let children = store.get_children(&[1u8; 32]).await;
        assert_eq!(children.len(), 2);
        assert!(children.contains(&[2u8; 32]));
        assert!(children.contains(&[3u8; 32]));
    }

    #[tokio::test]
    async fn test_cleanup() {
        let store = VertexStore::new_in_memory();

        // Store vertices across multiple rounds
        for round in 0..10 {
            let vertex = create_test_vertex(round as u8, round, 1, vec![]);
            store.store_vertex(vertex).await.unwrap();
        }

        let stats_before = store.get_stats().await;
        assert_eq!(stats_before.total_vertices, 10);

        // Keep only last 3 rounds (rounds 7, 8, 9)
        let removed = store.cleanup_old_vertices(3).await.unwrap();
        assert_eq!(removed, 7);

        let stats_after = store.get_stats().await;
        assert_eq!(stats_after.total_vertices, 3);
        assert_eq!(stats_after.earliest_round, 7);
        assert_eq!(stats_after.latest_round, 9);
    }
}
