use q_types::*;
use anyhow::Result;
use std::collections::{HashMap, BTreeMap};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// High-performance storage for DAG vertices
/// Maintains causal ordering and enables fast lookups
pub struct VertexStore {
    /// All vertices by ID
    vertices: RwLock<HashMap<VertexId, Vertex>>,
    /// Vertices organized by round
    vertices_by_round: RwLock<BTreeMap<Round, Vec<VertexId>>>,
    /// Vertices organized by author
    vertices_by_author: RwLock<HashMap<NodeId, Vec<VertexId>>>,
    /// Parent-child relationships for DAG traversal
    children: RwLock<HashMap<VertexId, Vec<VertexId>>>,
    /// Fast lookup for causal dependencies
    causal_history: RwLock<HashMap<VertexId, Vec<VertexId>>>,
}

impl VertexStore {
    pub fn new() -> Self {
        Self {
            vertices: RwLock::new(HashMap::new()),
            vertices_by_round: RwLock::new(BTreeMap::new()),
            vertices_by_author: RwLock::new(HashMap::new()),
            children: RwLock::new(HashMap::new()),
            causal_history: RwLock::new(HashMap::new()),
        }
    }

    /// Store a vertex with all index updates
    pub async fn store_vertex(&self, vertex: Vertex) -> Result<()> {
        let vertex_id = vertex.id;
        
        info!("Storing vertex {} from round {} by author {:?}", 
              hex::encode(vertex_id), vertex.round, vertex.author);

        // Store main vertex
        {
            let mut vertices = self.vertices.write().await;
            vertices.insert(vertex_id, vertex.clone());
        }

        // Update round index
        {
            let mut vertices_by_round = self.vertices_by_round.write().await;
            vertices_by_round
                .entry(vertex.round)
                .or_insert_with(Vec::new)
                .push(vertex_id);
        }

        // Update author index
        {
            let mut vertices_by_author = self.vertices_by_author.write().await;
            vertices_by_author
                .entry(vertex.author)
                .or_insert_with(Vec::new)
                .push(vertex_id);
        }

        // Update parent-child relationships
        {
            let mut children = self.children.write().await;
            for parent_id in &vertex.parents {
                children
                    .entry(*parent_id)
                    .or_insert_with(Vec::new)
                    .push(vertex_id);
            }
        }

        // Compute and store causal history
        self.update_causal_history(vertex_id, &vertex.parents).await?;

        debug!("Successfully stored vertex {} with {} parents", 
               hex::encode(vertex_id), vertex.parents.len());

        Ok(())
    }

    /// Get vertex by ID
    pub async fn get_vertex(&self, vertex_id: &VertexId) -> Option<Vertex> {
        let vertices = self.vertices.read().await;
        vertices.get(vertex_id).cloned()
    }

    /// Get all vertices in a round
    pub async fn get_vertices_in_round(&self, round: Round) -> Vec<Vertex> {
        let vertices = self.vertices.read().await;
        let vertices_by_round = self.vertices_by_round.read().await;
        
        if let Some(vertex_ids) = vertices_by_round.get(&round) {
            vertex_ids
                .iter()
                .filter_map(|id| vertices.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get vertices by author
    pub async fn get_vertices_by_author(&self, author: &NodeId) -> Vec<Vertex> {
        let vertices = self.vertices.read().await;
        let vertices_by_author = self.vertices_by_author.read().await;
        
        if let Some(vertex_ids) = vertices_by_author.get(author) {
            vertex_ids
                .iter()
                .filter_map(|id| vertices.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get children of a vertex
    pub async fn get_children(&self, vertex_id: &VertexId) -> Vec<VertexId> {
        let children = self.children.read().await;
        children.get(vertex_id).cloned().unwrap_or_else(Vec::new)
    }

    /// Check if vertex exists
    pub async fn contains_vertex(&self, vertex_id: &VertexId) -> bool {
        let vertices = self.vertices.read().await;
        vertices.contains_key(vertex_id)
    }

    /// Get causal history of a vertex (all ancestors)
    pub async fn get_causal_history(&self, vertex_id: &VertexId) -> Vec<VertexId> {
        let causal_history = self.causal_history.read().await;
        causal_history.get(vertex_id).cloned().unwrap_or_else(Vec::new)
    }

    /// Check if vertex A causally precedes vertex B
    pub async fn causally_precedes(&self, a: &VertexId, b: &VertexId) -> bool {
        let causal_history = self.causal_history.read().await;
        
        if let Some(history_b) = causal_history.get(b) {
            history_b.contains(a)
        } else {
            false
        }
    }

    /// Get latest round with vertices
    pub async fn get_latest_round(&self) -> Option<Round> {
        let vertices_by_round = self.vertices_by_round.read().await;
        vertices_by_round.keys().last().copied()
    }

    /// Get range of rounds [start, end]
    pub async fn get_round_range(&self, start: Round, end: Round) -> Vec<Vertex> {
        let vertices = self.vertices.read().await;
        let vertices_by_round = self.vertices_by_round.read().await;
        
        let mut result = Vec::new();
        for round in start..=end {
            if let Some(vertex_ids) = vertices_by_round.get(&round) {
                for vertex_id in vertex_ids {
                    if let Some(vertex) = vertices.get(vertex_id) {
                        result.push(vertex.clone());
                    }
                }
            }
        }
        
        result
    }

    /// Update causal history for a new vertex
    async fn update_causal_history(
        &self,
        vertex_id: VertexId,
        parents: &[VertexId],
    ) -> Result<()> {
        let mut causal_history = self.causal_history.write().await;
        let mut history = Vec::new();
        
        // Add all parents to history
        for parent_id in parents {
            history.push(*parent_id);
            
            // Add all ancestors of each parent
            if let Some(parent_history) = causal_history.get(parent_id) {
                for ancestor in parent_history {
                    if !history.contains(ancestor) {
                        history.push(*ancestor);
                    }
                }
            }
        }
        
        causal_history.insert(vertex_id, history);
        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> VertexStoreStats {
        let vertices = self.vertices.read().await;
        let vertices_by_round = self.vertices_by_round.read().await;
        let vertices_by_author = self.vertices_by_author.read().await;
        
        let total_vertices = vertices.len();
        let total_rounds = vertices_by_round.len();
        let total_authors = vertices_by_author.len();
        
        let latest_round = vertices_by_round.keys().last().copied().unwrap_or(0);
        let earliest_round = vertices_by_round.keys().next().copied().unwrap_or(0);
        
        let mut vertices_per_round = BTreeMap::new();
        for (round, vertex_ids) in vertices_by_round.iter() {
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

    /// Clean up old vertices beyond retention policy
    pub async fn cleanup_old_vertices(&self, keep_rounds: u64) -> Result<usize> {
        let latest_round = self.get_latest_round().await.unwrap_or(0);
        let cutoff_round = latest_round.saturating_sub(keep_rounds);
        
        let mut removed_count = 0;
        
        // Get vertices to remove
        let vertices_to_remove: Vec<VertexId> = {
            let vertices_by_round = self.vertices_by_round.read().await;
            let mut to_remove = Vec::new();
            
            for (round, vertex_ids) in vertices_by_round.iter() {
                if *round < cutoff_round {
                    to_remove.extend(vertex_ids.iter().copied());
                }
            }
            
            to_remove
        };
        
        // Remove from all indices
        for vertex_id in vertices_to_remove {
            // Get vertex info before removal
            let (round, author) = {
                let vertices = self.vertices.read().await;
                if let Some(vertex) = vertices.get(&vertex_id) {
                    (vertex.round, vertex.author)
                } else {
                    continue;
                }
            };
            
            // Remove from main storage
            {
                let mut vertices = self.vertices.write().await;
                vertices.remove(&vertex_id);
            }
            
            // Remove from round index
            {
                let mut vertices_by_round = self.vertices_by_round.write().await;
                if let Some(round_vertices) = vertices_by_round.get_mut(&round) {
                    round_vertices.retain(|id| *id != vertex_id);
                    if round_vertices.is_empty() {
                        vertices_by_round.remove(&round);
                    }
                }
            }
            
            // Remove from author index
            {
                let mut vertices_by_author = self.vertices_by_author.write().await;
                if let Some(author_vertices) = vertices_by_author.get_mut(&author) {
                    author_vertices.retain(|id| *id != vertex_id);
                    if author_vertices.is_empty() {
                        vertices_by_author.remove(&author);
                    }
                }
            }
            
            // Remove from children index
            {
                let mut children = self.children.write().await;
                children.remove(&vertex_id);
            }
            
            // Remove from causal history
            {
                let mut causal_history = self.causal_history.write().await;
                causal_history.remove(&vertex_id);
            }
            
            removed_count += 1;
        }
        
        info!("Cleaned up {} vertices older than round {}", removed_count, cutoff_round);
        Ok(removed_count)
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
        let store = VertexStore::new();
        let vertex = create_test_vertex(1, 0, 1, vec![]);
        
        store.store_vertex(vertex.clone()).await.unwrap();
        
        let retrieved = store.get_vertex(&[1u8; 32]).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, vertex.id);
    }

    #[tokio::test]
    async fn test_round_queries() {
        let store = VertexStore::new();
        
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
        let store = VertexStore::new();
        
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
        let store = VertexStore::new();
        
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
        let store = VertexStore::new();
        
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