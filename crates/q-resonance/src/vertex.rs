//! Resonance Vertex for hypergraph DAG
//!
//! Extends traditional DAG vertices with multi-dimensional coordinates and string states

use crate::string_state::StringState;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Multi-dimensional coordinates for hypergraph embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphCoordinates {
    /// Temporal dimension (round number, causal ordering)
    pub temporal: f64,

    /// Spatial dimension (network topology, RTT-based)
    pub spatial: Vec<f64>,

    /// Energetic dimension (stake weight, transaction fees)
    pub energetic: f64,

    /// Entropic dimension (quantum randomness from VDF)
    pub entropic: f64,

    /// Metadata dimensions (ZK-proofs, oracle data as gauge fields)
    pub metadata: HashMap<String, f64>,
}

impl HypergraphCoordinates {
    /// Create coordinates from basic parameters
    pub fn new(
        round: u64,
        network_position: Vec<f64>,
        stake: f64,
        entropy: f64,
    ) -> Self {
        Self {
            temporal: round as f64,
            spatial: network_position,
            energetic: stake,
            entropic: entropy,
            metadata: HashMap::new(),
        }
    }

    /// Compute distance to another coordinate in hypergraph space
    pub fn distance(&self, other: &HypergraphCoordinates) -> f64 {
        let temporal_dist = (self.temporal - other.temporal).powi(2);

        let spatial_dist: f64 = self
            .spatial
            .iter()
            .zip(&other.spatial)
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let energetic_dist = (self.energetic - other.energetic).powi(2);
        let entropic_dist = (self.entropic - other.entropic).powi(2);

        (temporal_dist + spatial_dist + energetic_dist + entropic_dist).sqrt()
    }

    /// Add metadata dimension
    pub fn add_metadata(&mut self, key: String, value: f64) {
        self.metadata.insert(key, value);
    }
}

/// Resonance Vertex in the hypergraph DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceVertex {
    /// Unique vertex hash
    pub hash: [u8; 32],

    /// Round number in consensus
    pub round: u64,

    /// Parent vertices (strong edges in DAG)
    pub parents: HashSet<[u8; 32]>,

    /// String state for this vertex
    pub string_state: StringState,

    /// Multi-dimensional coordinates
    pub coordinates: HypergraphCoordinates,

    /// Transaction payload
    pub transactions: Vec<Vec<u8>>,

    /// Signature (post-quantum: Dilithium)
    pub signature: Vec<u8>,

    /// Validator ID who created this vertex
    pub validator: Vec<u8>,

    /// Timestamp
    pub timestamp: u64,

    /// Resonance score (computed during ordering)
    pub resonance_score: f64,
}

impl ResonanceVertex {
    /// Create a new resonance vertex
    pub fn new(
        hash: [u8; 32],
        round: u64,
        parents: HashSet<[u8; 32]>,
        transactions: Vec<Vec<u8>>,
        validator: Vec<u8>,
        timestamp: u64,
        stake: f64,
        network_position: Vec<f64>,
        entropy: f64,
    ) -> Self {
        let coordinates = HypergraphCoordinates::new(round, network_position.clone(), stake, entropy);

        // Create string state from vertex parameters
        let priority = 1.0 / (round as f64 + 1.0); // Earlier rounds have higher priority
        let string_state = StringState::new(
            stake,
            priority,
            network_position,
            hash,
            timestamp,
        );

        Self {
            hash,
            round,
            parents,
            string_state,
            coordinates,
            transactions,
            signature: vec![],
            validator,
            timestamp,
            resonance_score: 0.0,
        }
    }

    /// Compute resonance with another vertex
    pub fn resonance(&self, other: &ResonanceVertex) -> f64 {
        // String state resonance
        let string_resonance = self.string_state.resonance(&other.string_state);

        // Spatial proximity bonus
        let distance = self.coordinates.distance(&other.coordinates);
        let proximity_factor = (-distance / 10.0).exp();

        // Temporal ordering preservation
        let temporal_factor = if self.round < other.round {
            1.0
        } else if self.round == other.round {
            0.5
        } else {
            0.1 // Penalize reverse causality
        };

        string_resonance * proximity_factor * temporal_factor
    }

    /// Check if this vertex causally precedes another
    pub fn precedes(&self, other: &ResonanceVertex) -> bool {
        // Check temporal ordering
        if self.round >= other.round {
            return false;
        }

        // Check if this vertex is in other's causal past
        other.is_ancestor(&self.hash)
    }

    /// Check if a vertex is an ancestor (BFS through parents)
    pub fn is_ancestor(&self, ancestor_hash: &[u8; 32]) -> bool {
        let mut visited = HashSet::new();
        let mut queue: Vec<[u8; 32]> = self.parents.iter().copied().collect();

        while let Some(current) = queue.pop() {
            if &current == ancestor_hash {
                return true;
            }

            if visited.insert(current) {
                // In real implementation, lookup parent vertices from DAG
                // For now, just check direct parents
            }
        }

        false
    }

    /// Compute hash of vertex content
    pub fn compute_hash(&self) -> [u8; 32] {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&self.round.to_le_bytes());

        for parent in &self.parents {
            hasher.update(parent);
        }

        for tx in &self.transactions {
            hasher.update(tx);
        }

        hasher.update(&self.validator);
        hasher.update(&self.timestamp.to_le_bytes());

        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_bytes());
        result
    }

    /// Sign vertex with post-quantum signature
    pub fn sign(&mut self, _private_key: &[u8]) {
        // In production, use Dilithium5 signature
        // self.signature = dilithium_sign(self.compute_hash(), private_key);
        self.signature = vec![0u8; 64]; // Placeholder
    }

    /// Verify vertex signature
    pub fn verify_signature(&self, _public_key: &[u8]) -> bool {
        // In production, use Dilithium5 verification
        // dilithium_verify(self.compute_hash(), &self.signature, public_key)
        true // Placeholder
    }

    /// Update resonance score
    pub fn update_resonance_score(&mut self, score: f64) {
        self.resonance_score = score;
    }

    /// Check if vertex should be ordered before another
    pub fn should_order_before(&self, other: &ResonanceVertex) -> bool {
        // Primary: Temporal ordering (earlier rounds first)
        if self.round != other.round {
            return self.round < other.round;
        }

        // Secondary: Resonance score (higher resonance first)
        if (self.resonance_score - other.resonance_score).abs() > 1e-6 {
            return self.resonance_score > other.resonance_score;
        }

        // Tertiary: Hash comparison (deterministic tie-breaking)
        self.hash < other.hash
    }
}

/// Causal DAG structure for vertices
#[derive(Debug)]
pub struct CausalDAG {
    /// All vertices indexed by hash
    vertices: HashMap<[u8; 32], ResonanceVertex>,

    /// Adjacency list (parent -> children)
    children: HashMap<[u8; 32], HashSet<[u8; 32]>>,

    /// Topologically ordered vertices (updated on consensus)
    ordered: Vec<[u8; 32]>,
}

impl CausalDAG {
    /// Create a new empty DAG
    pub fn new() -> Self {
        Self {
            vertices: HashMap::new(),
            children: HashMap::new(),
            ordered: Vec::new(),
        }
    }

    /// Add a vertex to the DAG
    pub fn add_vertex(&mut self, vertex: ResonanceVertex) {
        let hash = vertex.hash;

        // Add edges from parents to this vertex
        for parent in &vertex.parents {
            self.children
                .entry(*parent)
                .or_insert_with(HashSet::new)
                .insert(hash);
        }

        self.vertices.insert(hash, vertex);
    }

    /// Check if there's a directed edge from src to dst
    pub fn has_edge(&self, src: [u8; 32], dst: [u8; 32]) -> bool {
        self.children
            .get(&src)
            .map(|children| children.contains(&dst))
            .unwrap_or(false)
    }

    /// Get vertex by hash
    pub fn get_vertex(&self, hash: &[u8; 32]) -> Option<&ResonanceVertex> {
        self.vertices.get(hash)
    }

    /// Get mutable vertex by hash
    pub fn get_vertex_mut(&mut self, hash: &[u8; 32]) -> Option<&mut ResonanceVertex> {
        self.vertices.get_mut(hash)
    }

    /// Get all vertices in current round
    pub fn vertices_in_round(&self, round: u64) -> Vec<&ResonanceVertex> {
        self.vertices
            .values()
            .filter(|v| v.round == round)
            .collect()
    }

    /// Topologically sort vertices (for total ordering)
    pub fn topological_sort(&mut self) {
        // Kahn's algorithm for topological sorting
        let mut in_degree: HashMap<[u8; 32], usize> = HashMap::new();

        // Calculate in-degrees
        for vertex in self.vertices.values() {
            in_degree.entry(vertex.hash).or_insert(0);
            for child in self.children.get(&vertex.hash).unwrap_or(&HashSet::new()) {
                *in_degree.entry(*child).or_insert(0) += 1;
            }
        }

        // Queue of vertices with in-degree 0
        let mut queue: Vec<[u8; 32]> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(hash, _)| *hash)
            .collect();

        let mut result = Vec::new();

        while let Some(hash) = queue.pop() {
            result.push(hash);

            if let Some(children) = self.children.get(&hash) {
                for child in children {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(*child);
                        }
                    }
                }
            }
        }

        self.ordered = result;
    }

    /// Get ordered vertices
    pub fn get_ordered(&self) -> &Vec<[u8; 32]> {
        &self.ordered
    }
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergraph_coordinates() {
        let coord1 = HypergraphCoordinates::new(1, vec![0.0, 0.0], 100.0, 0.5);
        let coord2 = HypergraphCoordinates::new(2, vec![1.0, 1.0], 110.0, 0.6);

        let distance = coord1.distance(&coord2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_resonance_vertex_creation() {
        let hash = [1u8; 32];
        let parents = HashSet::new();
        let vertex = ResonanceVertex::new(
            hash,
            1,
            parents,
            vec![],
            vec![1, 2, 3],
            1000,
            100.0,
            vec![0.0, 0.0],
            0.5,
        );

        assert_eq!(vertex.round, 1);
        assert_eq!(vertex.hash, hash);
    }

    #[test]
    fn test_vertex_resonance() {
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];

        let vertex1 = ResonanceVertex::new(
            hash1,
            1,
            HashSet::new(),
            vec![],
            vec![1],
            1000,
            100.0,
            vec![0.0, 0.0],
            0.5,
        );

        let vertex2 = ResonanceVertex::new(
            hash2,
            2,
            HashSet::new(),
            vec![],
            vec![2],
            1001,
            100.0,
            vec![0.1, 0.1],
            0.5,
        );

        let resonance = vertex1.resonance(&vertex2);
        assert!(resonance >= 0.0);
    }

    #[test]
    fn test_causal_dag() {
        let mut dag = CausalDAG::new();

        let hash1 = [1u8; 32];
        let vertex1 = ResonanceVertex::new(
            hash1,
            1,
            HashSet::new(),
            vec![],
            vec![1],
            1000,
            100.0,
            vec![0.0],
            0.5,
        );

        dag.add_vertex(vertex1);

        let mut parents = HashSet::new();
        parents.insert(hash1);

        let hash2 = [2u8; 32];
        let vertex2 = ResonanceVertex::new(
            hash2,
            2,
            parents,
            vec![],
            vec![2],
            1001,
            100.0,
            vec![0.0],
            0.5,
        );

        dag.add_vertex(vertex2);

        assert!(dag.has_edge(hash1, hash2));
        assert_eq!(dag.vertices_in_round(1).len(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = CausalDAG::new();

        // Create a simple DAG: v1 -> v2 -> v3
        let hash1 = [1u8; 32];
        let vertex1 = ResonanceVertex::new(
            hash1,
            1,
            HashSet::new(),
            vec![],
            vec![1],
            1000,
            100.0,
            vec![0.0],
            0.5,
        );
        dag.add_vertex(vertex1);

        let hash2 = [2u8; 32];
        let mut parents2 = HashSet::new();
        parents2.insert(hash1);
        let vertex2 = ResonanceVertex::new(
            hash2,
            2,
            parents2,
            vec![],
            vec![2],
            1001,
            100.0,
            vec![0.0],
            0.5,
        );
        dag.add_vertex(vertex2);

        let hash3 = [3u8; 32];
        let mut parents3 = HashSet::new();
        parents3.insert(hash2);
        let vertex3 = ResonanceVertex::new(
            hash3,
            3,
            parents3,
            vec![],
            vec![3],
            1002,
            100.0,
            vec![0.0],
            0.5,
        );
        dag.add_vertex(vertex3);

        dag.topological_sort();

        let ordered = dag.get_ordered();
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0], hash1);
    }
}
