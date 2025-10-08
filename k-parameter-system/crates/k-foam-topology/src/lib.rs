//! Quantum foam topology analyzer
//! Network-based spacetime foam structure with Planck-scale fluctuations

use k_constants::{PhysicalConstants, PlanckScales};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum foam node (spacetime point)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoamNode {
    /// Node ID
    pub id: usize,
    /// Position in 3D space (x, y, z) in Planck units
    pub position: [f64; 3],
    /// Curvature scalar R at this point
    pub curvature: f64,
    /// Topological charge
    pub topological_charge: i32,
}

/// Quantum foam edge (spacetime connection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoamEdge {
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Connection strength (0.0 to 1.0)
    pub strength: f64,
    /// Causal structure (-1: timelike, 0: null, 1: spacelike)
    pub causality: i8,
}

/// Quantum foam network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFoamTopology {
    /// Nodes in the foam
    pub nodes: Vec<FoamNode>,
    /// Edges connecting nodes
    pub edges: Vec<FoamEdge>,
    /// Planck scale reference
    pub planck_length: f64,
}

impl QuantumFoamTopology {
    /// Create new quantum foam with n nodes
    pub fn new(n_nodes: usize, planck: &PlanckScales) -> Self {
        let mut nodes = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            nodes.push(FoamNode {
                id: i,
                position: [
                    (i as f64).sin() * 10.0,
                    (i as f64).cos() * 10.0,
                    (i as f64 * 0.5).sin() * 10.0,
                ],
                curvature: 0.0,
                topological_charge: 0,
            });
        }

        Self {
            nodes,
            edges: Vec::new(),
            planck_length: planck.length,
        }
    }

    /// Add edge between two nodes
    pub fn add_edge(&mut self, source: usize, target: usize, strength: f64) {
        // Determine causality based on spatial separation
        let dx = self.distance(source, target);
        let causality = if dx < self.planck_length {
            0 // Null-like at Planck scale
        } else if dx < 2.0 * self.planck_length {
            -1 // Timelike
        } else {
            1 // Spacelike
        };

        self.edges.push(FoamEdge {
            source,
            target,
            strength,
            causality,
        });
    }

    /// Calculate Euclidean distance between nodes
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let p1 = &self.nodes[i].position;
        let p2 = &self.nodes[j].position;

        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
            * self.planck_length
    }

    /// Calculate adjacency matrix
    pub fn adjacency_matrix(&self) -> Array2<f64> {
        let n = self.nodes.len();
        let mut matrix = Array2::zeros((n, n));

        for edge in &self.edges {
            matrix[[edge.source, edge.target]] = edge.strength;
            matrix[[edge.target, edge.source]] = edge.strength; // Symmetric
        }

        matrix
    }

    /// Calculate node degree centrality
    pub fn degree_centrality(&self, node_id: usize) -> f64 {
        self.edges
            .iter()
            .filter(|e| e.source == node_id || e.target == node_id)
            .map(|e| e.strength)
            .sum()
    }

    /// Compute topological invariant (Euler characteristic approximation)
    pub fn euler_characteristic(&self) -> i32 {
        let v = self.nodes.len() as i32; // Vertices
        let e = self.edges.len() as i32; // Edges
        // χ = V - E + F (for 2D surface, simplified)
        // For quantum foam, we approximate with χ ≈ V - E
        v - e
    }

    /// Generate random foam connections (Erdős-Rényi graph)
    pub fn generate_random_connections(&mut self, connection_probability: f64) {
        let n = self.nodes.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if rand_uniform() < connection_probability {
                    let strength = 0.5 + 0.5 * rand_uniform();
                    self.add_edge(i, j, strength);
                }
            }
        }
    }

    /// Calculate foam fluctuation amplitude
    pub fn fluctuation_amplitude(&self) -> f64 {
        // Average curvature variance
        let mean_curvature: f64 = self.nodes.iter().map(|n| n.curvature).sum::<f64>() / self.nodes.len() as f64;
        let variance: f64 = self.nodes
            .iter()
            .map(|n| (n.curvature - mean_curvature).powi(2))
            .sum::<f64>()
            / self.nodes.len() as f64;

        variance.sqrt()
    }
}

/// Simple pseudo-random number generator (for deterministic testing)
fn rand_uniform() -> f64 {
    // Using a simple deterministic sequence for reproducibility
    // In production, use proper RNG like `rand` crate
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED / 65536) as f64 % 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foam_creation() {
        let constants = PhysicalConstants::default();
        let planck = PlanckScales::from_constants(&constants);
        let foam = QuantumFoamTopology::new(10, &planck);

        assert_eq!(foam.nodes.len(), 10);
        assert!(foam.planck_length > 0.0);
    }

    #[test]
    fn test_foam_edges() {
        let constants = PhysicalConstants::default();
        let planck = PlanckScales::from_constants(&constants);
        let mut foam = QuantumFoamTopology::new(5, &planck);

        foam.add_edge(0, 1, 0.8);
        foam.add_edge(1, 2, 0.6);

        assert_eq!(foam.edges.len(), 2);
        assert_eq!(foam.degree_centrality(1), 1.4);
    }

    #[test]
    fn test_adjacency_matrix() {
        let constants = PhysicalConstants::default();
        let planck = PlanckScales::from_constants(&constants);
        let mut foam = QuantumFoamTopology::new(3, &planck);

        foam.add_edge(0, 1, 1.0);
        foam.add_edge(1, 2, 0.5);

        let adj = foam.adjacency_matrix();
        assert_eq!(adj[[0, 1]], 1.0);
        assert_eq!(adj[[1, 2]], 0.5);
    }

    #[test]
    fn test_euler_characteristic() {
        let constants = PhysicalConstants::default();
        let planck = PlanckScales::from_constants(&constants);
        let mut foam = QuantumFoamTopology::new(4, &planck);

        foam.add_edge(0, 1, 1.0);
        foam.add_edge(1, 2, 1.0);
        foam.add_edge(2, 3, 1.0);

        let chi = foam.euler_characteristic();
        assert_eq!(chi, 1); // 4 vertices - 3 edges = 1
    }
}
