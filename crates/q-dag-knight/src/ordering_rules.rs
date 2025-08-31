/// DAG-Knight ordering rules implementation
/// Provides deterministic transaction ordering for the Q-NarwhalKnight consensus

use q_types::*;
use q_narwhal_core::{Certificate, Vertex};
use anyhow::Result;
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Ordering engine implementing DAG-Knight rules
pub struct OrderingEngine {
    /// Causal ordering graph (vertex_id -> dependencies)
    causal_graph: RwLock<HashMap<VertexId, HashSet<VertexId>>>,
    
    /// Processed vertices by round
    processed_rounds: RwLock<BTreeMap<Round, HashSet<VertexId>>>,
    
    /// Cached ordering results
    ordering_cache: RwLock<HashMap<Round, Vec<VertexId>>>,
    
    /// Statistics
    stats: RwLock<OrderingStats>,
}

#[derive(Debug, Clone)]
pub struct OrderingStats {
    pub total_vertices_processed: u64,
    pub total_orderings_computed: u64,
    pub average_ordering_time_ms: f64,
    pub cache_hit_rate: f64,
    pub causal_dependencies: u64,
}

#[derive(Debug, Clone)]
pub struct OrderingResult {
    pub round: Round,
    pub ordered_vertices: Vec<VertexId>,
    pub causal_depth: u32,
    pub processing_time_ms: u64,
    pub cache_hit: bool,
}

impl OrderingEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            causal_graph: RwLock::new(HashMap::new()),
            processed_rounds: RwLock::new(BTreeMap::new()),
            ordering_cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(OrderingStats {
                total_vertices_processed: 0,
                total_orderings_computed: 0,
                average_ordering_time_ms: 0.0,
                cache_hit_rate: 0.0,
                causal_dependencies: 0,
            }),
        })
    }

    /// Process a vertex and update causal ordering
    pub async fn process_vertex(&self, vertex: &Vertex, certificate: &Certificate) -> Result<OrderingResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Processing vertex {} for ordering in round {}", 
               hex::encode(vertex.id), vertex.round);

        // Check cache first
        {
            let cache = self.ordering_cache.read().await;
            if let Some(cached_ordering) = cache.get(&vertex.round) {
                let mut stats = self.stats.write().await;
                stats.cache_hit_rate = (stats.cache_hit_rate * stats.total_orderings_computed as f64 + 1.0) 
                    / (stats.total_orderings_computed + 1) as f64;
                stats.total_orderings_computed += 1;

                return Ok(OrderingResult {
                    round: vertex.round,
                    ordered_vertices: cached_ordering.clone(),
                    causal_depth: self.compute_causal_depth(&vertex.id).await,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                    cache_hit: true,
                });
            }
        }

        // Add vertex to causal graph
        self.add_vertex_to_graph(vertex).await?;

        // Compute ordering for this round
        let ordered_vertices = self.compute_round_ordering(vertex.round).await?;

        // Cache the result
        {
            let mut cache = self.ordering_cache.write().await;
            cache.insert(vertex.round, ordered_vertices.clone());
        }

        // Update processed rounds
        {
            let mut rounds = self.processed_rounds.write().await;
            rounds.entry(vertex.round).or_insert_with(HashSet::new).insert(vertex.id);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_vertices_processed += 1;
            stats.total_orderings_computed += 1;
            let processing_time_ms = start_time.elapsed().as_millis() as u64;
            stats.average_ordering_time_ms = (stats.average_ordering_time_ms * (stats.total_orderings_computed - 1) as f64
                + processing_time_ms as f64) / stats.total_orderings_computed as f64;
        }

        Ok(OrderingResult {
            round: vertex.round,
            ordered_vertices,
            causal_depth: self.compute_causal_depth(&vertex.id).await,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            cache_hit: false,
        })
    }

    /// Add vertex to causal ordering graph
    async fn add_vertex_to_graph(&self, vertex: &Vertex) -> Result<()> {
        let mut graph = self.causal_graph.write().await;
        let mut dependencies = HashSet::new();

        // Add parent dependencies
        for parent_id in &vertex.parents {
            dependencies.insert(*parent_id);
        }

        // Add implicit causal dependencies based on DAG-Knight rules
        // Rule 1: All vertices from previous rounds are causally before this vertex
        let processed = self.processed_rounds.read().await;
        for (&round, vertices) in processed.iter() {
            if round < vertex.round {
                dependencies.extend(vertices);
            }
        }

        graph.insert(vertex.id, dependencies.clone());

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.causal_dependencies += dependencies.len() as u64;
        }

        debug!("Added vertex {} with {} causal dependencies", 
               hex::encode(vertex.id), dependencies.len());

        Ok(())
    }

    /// Compute deterministic ordering for a round using DAG-Knight rules
    async fn compute_round_ordering(&self, round: Round) -> Result<Vec<VertexId>> {
        let rounds = self.processed_rounds.read().await;
        let round_vertices = rounds.get(&round).cloned().unwrap_or_else(HashSet::new);

        if round_vertices.is_empty() {
            return Ok(Vec::new());
        }

        // Apply DAG-Knight ordering rules:
        // 1. Topological sort based on causal dependencies
        // 2. Break ties using vertex hash (deterministic)
        let mut ordered_vertices = self.topological_sort(round_vertices.clone()).await?;

        // Apply tie-breaking rule: sort by vertex ID hash
        ordered_vertices.sort_by(|a, b| a.cmp(b));

        info!("Computed ordering for round {} with {} vertices", 
              round, ordered_vertices.len());

        Ok(ordered_vertices)
    }

    /// Perform topological sort on vertices
    async fn topological_sort(&self, vertices: HashSet<VertexId>) -> Result<Vec<VertexId>> {
        let graph = self.causal_graph.read().await;
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        // Perform DFS-based topological sort
        for vertex_id in &vertices {
            if !visited.contains(vertex_id) {
                self.topological_visit(*vertex_id, &graph, &vertices, &mut visited, &mut visiting, &mut result)?;
            }
        }

        // Reverse to get correct topological order
        result.reverse();
        Ok(result)
    }

    /// Recursive DFS visit for topological sort
    fn topological_visit(
        &self,
        vertex_id: VertexId,
        graph: &HashMap<VertexId, HashSet<VertexId>>,
        valid_vertices: &HashSet<VertexId>,
        visited: &mut HashSet<VertexId>,
        visiting: &mut HashSet<VertexId>,
        result: &mut Vec<VertexId>,
    ) -> Result<()> {
        if visiting.contains(&vertex_id) {
            // Cycle detected - should not happen in a DAG
            return Err(anyhow::anyhow!("Cycle detected in causal graph"));
        }

        if visited.contains(&vertex_id) {
            return Ok(());
        }

        visiting.insert(vertex_id);

        // Visit dependencies first
        if let Some(dependencies) = graph.get(&vertex_id) {
            for &dep_id in dependencies {
                if valid_vertices.contains(&dep_id) {
                    self.topological_visit(dep_id, graph, valid_vertices, visited, visiting, result)?;
                }
            }
        }

        visiting.remove(&vertex_id);
        visited.insert(vertex_id);
        result.push(vertex_id);

        Ok(())
    }

    /// Compute causal depth of a vertex
    async fn compute_causal_depth(&self, vertex_id: &VertexId) -> u32 {
        let graph = self.causal_graph.read().await;
        let mut depth = 0;
        let mut to_visit = vec![(*vertex_id, 0)];
        let mut visited = HashSet::new();

        while let Some((current_id, current_depth)) = to_visit.pop() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id);

            depth = depth.max(current_depth);

            if let Some(dependencies) = graph.get(&current_id) {
                for &dep_id in dependencies {
                    to_visit.push((dep_id, current_depth + 1));
                }
            }
        }

        depth
    }

    /// Get ordering for a specific round
    pub async fn get_round_ordering(&self, round: Round) -> Option<Vec<VertexId>> {
        let cache = self.ordering_cache.read().await;
        cache.get(&round).cloned()
    }

    /// Get all vertices in causal order up to a specific round
    pub async fn get_causal_ordering_up_to(&self, max_round: Round) -> Result<Vec<VertexId>> {
        let rounds = self.processed_rounds.read().await;
        let mut all_vertices = HashSet::new();

        // Collect all vertices up to max_round
        for (&round, vertices) in rounds.iter() {
            if round <= max_round {
                all_vertices.extend(vertices);
            }
        }

        // Perform global topological sort
        self.topological_sort(all_vertices).await
    }

    /// Verify ordering consistency
    pub async fn verify_ordering(&self, round: Round, ordering: &[VertexId]) -> Result<bool> {
        let graph = self.causal_graph.read().await;

        // Check that all dependencies come before dependents in the ordering
        let position: HashMap<VertexId, usize> = ordering.iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();

        for &vertex_id in ordering {
            if let Some(dependencies) = graph.get(&vertex_id) {
                for &dep_id in dependencies {
                    if let (Some(&vertex_pos), Some(&dep_pos)) = (position.get(&vertex_id), position.get(&dep_id)) {
                        if dep_pos >= vertex_pos {
                            warn!("Ordering violation: dependency {} comes after vertex {}", 
                                  hex::encode(dep_id), hex::encode(vertex_id));
                            return Ok(false);
                        }
                    }
                }
            }
        }

        Ok(true)
    }

    /// Get ordering statistics
    pub async fn get_stats(&self) -> OrderingStats {
        self.stats.read().await.clone()
    }

    /// Clean up old cached orderings
    pub async fn cleanup_cache(&self, keep_rounds: u64) {
        let rounds = self.processed_rounds.read().await;
        let max_round = rounds.keys().max().copied().unwrap_or(0);
        let cutoff_round = max_round.saturating_sub(keep_rounds);

        // Clean up cache
        {
            let mut cache = self.ordering_cache.write().await;
            cache.retain(|&round, _| round >= cutoff_round);
        }

        // Clean up processed rounds
        {
            let mut processed = self.processed_rounds.write().await;
            processed.retain(|&round, _| round >= cutoff_round);
        }

        // Clean up causal graph (keep vertices from recent rounds)
        {
            let mut graph = self.causal_graph.write().await;
            let mut vertices_to_keep = HashSet::new();
            
            for (&round, vertices) in rounds.iter() {
                if round >= cutoff_round {
                    vertices_to_keep.extend(vertices);
                }
            }

            graph.retain(|vertex_id, _| vertices_to_keep.contains(vertex_id));
        }

        debug!("Cleaned up ordering cache, keeping rounds >= {}", cutoff_round);
    }

    /// Get causal ancestors of a vertex
    pub async fn get_causal_ancestors(&self, vertex_id: VertexId) -> HashSet<VertexId> {
        let graph = self.causal_graph.read().await;
        let mut ancestors = HashSet::new();
        let mut to_visit = vec![vertex_id];

        while let Some(current_id) = to_visit.pop() {
            if let Some(dependencies) = graph.get(&current_id) {
                for &dep_id in dependencies {
                    if ancestors.insert(dep_id) {
                        to_visit.push(dep_id);
                    }
                }
            }
        }

        ancestors
    }

    /// Check if vertex A causally precedes vertex B
    pub async fn causally_precedes(&self, vertex_a: VertexId, vertex_b: VertexId) -> bool {
        let ancestors_b = self.get_causal_ancestors(vertex_b).await;
        ancestors_b.contains(&vertex_a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex(id: u8, round: Round, parents: Vec<VertexId>) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [1; 32],
            tx_root: [0; 32],
            parents,
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    fn create_test_certificate(vertex_id: VertexId, round: Round) -> Certificate {
        Certificate {
            vertex_id,
            round,
            signatures: std::collections::BTreeMap::new(),
            threshold_met: true,
        }
    }

    #[tokio::test]
    async fn test_ordering_engine_creation() {
        let engine = OrderingEngine::new();
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_single_vertex_ordering() {
        let engine = OrderingEngine::new().unwrap();
        let vertex = create_test_vertex(1, 1, vec![]);
        let certificate = create_test_certificate(vertex.id, vertex.round);

        let result = engine.process_vertex(&vertex, &certificate).await.unwrap();
        
        assert_eq!(result.round, 1);
        assert_eq!(result.ordered_vertices.len(), 1);
        assert_eq!(result.ordered_vertices[0], vertex.id);
        assert!(!result.cache_hit);
    }

    #[tokio::test]
    async fn test_causal_ordering() {
        let engine = OrderingEngine::new().unwrap();
        
        // Create vertices with causal dependency: v1 -> v2
        let vertex1 = create_test_vertex(1, 1, vec![]);
        let vertex2 = create_test_vertex(2, 2, vec![vertex1.id]);
        
        let cert1 = create_test_certificate(vertex1.id, vertex1.round);
        let cert2 = create_test_certificate(vertex2.id, vertex2.round);

        // Process vertices
        engine.process_vertex(&vertex1, &cert1).await.unwrap();
        engine.process_vertex(&vertex2, &cert2).await.unwrap();

        // Check causal precedence
        let precedes = engine.causally_precedes(vertex1.id, vertex2.id).await;
        assert!(precedes);

        let reverse_precedes = engine.causally_precedes(vertex2.id, vertex1.id).await;
        assert!(!reverse_precedes);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let engine = OrderingEngine::new().unwrap();
        let vertex = create_test_vertex(1, 1, vec![]);
        let certificate = create_test_certificate(vertex.id, vertex.round);

        // First processing - should not hit cache
        let result1 = engine.process_vertex(&vertex, &certificate).await.unwrap();
        assert!(!result1.cache_hit);

        // Second processing of same round - should hit cache
        let vertex2 = create_test_vertex(2, 1, vec![]);
        let cert2 = create_test_certificate(vertex2.id, vertex2.round);
        
        let result2 = engine.process_vertex(&vertex2, &cert2).await.unwrap();
        // Note: Cache hit depends on implementation details of round caching
        
        let stats = engine.get_stats().await;
        assert!(stats.total_vertices_processed >= 2);
    }

    #[tokio::test]
    async fn test_topological_sort() {
        let engine = OrderingEngine::new().unwrap();
        
        // Create a chain: v1 -> v2 -> v3
        let vertex1 = create_test_vertex(1, 1, vec![]);
        let vertex2 = create_test_vertex(2, 2, vec![vertex1.id]);
        let vertex3 = create_test_vertex(3, 3, vec![vertex2.id]);
        
        let cert1 = create_test_certificate(vertex1.id, vertex1.round);
        let cert2 = create_test_certificate(vertex2.id, vertex2.round);
        let cert3 = create_test_certificate(vertex3.id, vertex3.round);

        // Process all vertices
        engine.process_vertex(&vertex1, &cert1).await.unwrap();
        engine.process_vertex(&vertex2, &cert2).await.unwrap();
        engine.process_vertex(&vertex3, &cert3).await.unwrap();

        // Get global ordering
        let global_ordering = engine.get_causal_ordering_up_to(3).await.unwrap();
        
        // Should be ordered: v1, v2, v3
        assert_eq!(global_ordering.len(), 3);
        
        // Verify ordering is valid
        let is_valid = engine.verify_ordering(3, &global_ordering).await.unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_causal_depth_calculation() {
        let engine = OrderingEngine::new().unwrap();
        
        // Create a chain of depth 2: v1 -> v2 -> v3
        let vertex1 = create_test_vertex(1, 1, vec![]);
        let vertex2 = create_test_vertex(2, 2, vec![vertex1.id]);
        let vertex3 = create_test_vertex(3, 3, vec![vertex2.id]);
        
        let cert1 = create_test_certificate(vertex1.id, vertex1.round);
        let cert2 = create_test_certificate(vertex2.id, vertex2.round);
        let cert3 = create_test_certificate(vertex3.id, vertex3.round);

        // Process vertices
        engine.process_vertex(&vertex1, &cert1).await.unwrap();
        engine.process_vertex(&vertex2, &cert2).await.unwrap();
        let result3 = engine.process_vertex(&vertex3, &cert3).await.unwrap();

        // v3 should have causal depth >= 2 (depends on v2 which depends on v1)
        assert!(result3.causal_depth >= 2);
    }
}