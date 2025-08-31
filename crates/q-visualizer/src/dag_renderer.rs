/// DAG visualization with quantum entanglement patterns
/// Creates Moiré interference patterns from quantum entropy correlations

use q_types::*;
use crate::{QuantumVertexExt, QuantumColorPalette};
use anyhow::Result;
use nalgebra::{Point2, Vector2};
use palette::{FromColor, Hsl, Srgb};
use plotters::prelude::*;
use std::collections::HashMap;
use svg::Document;
use svg::node::element::{Circle, Line, Group, Text};
use tokio::sync::RwLock;

/// High-performance DAG renderer with quantum aesthetic patterns
#[derive(Clone)]
pub struct DAGRenderer {
    vertices: RwLock<HashMap<VertexId, PositionedVertex>>,
    edges: RwLock<Vec<QuantumEdge>>,
    canvas_size: (u32, u32),
    color_palette: QuantumColorPalette,
    layout_engine: SpringLayoutEngine,
}

#[derive(Debug, Clone)]
pub struct PositionedVertex {
    pub vertex: Vertex,
    pub position: Point2<f64>,
    pub velocity: Vector2<f64>,
    pub quantum_hue: f32,
    pub entanglement_radius: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumEdge {
    pub from: VertexId,
    pub to: VertexId,
    pub strength: f64,
    pub phase_shift: f64,
    pub interference_pattern: Vec<Point2<f64>>,
}

/// Spring-force layout engine for natural DAG positioning
pub struct SpringLayoutEngine {
    spring_constant: f64,
    damping: f64,
    repulsion_strength: f64,
    attraction_strength: f64,
    ideal_edge_length: f64,
}

impl DAGRenderer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            vertices: RwLock::new(HashMap::new()),
            edges: RwLock::new(Vec::new()),
            canvas_size: (1920, 1080),
            color_palette: QuantumColorPalette::default(),
            layout_engine: SpringLayoutEngine::new(),
        })
    }

    /// Add a vertex to the DAG and compute its quantum properties
    pub async fn add_vertex(&self, vertex: &Vertex) -> Result<()> {
        let position = self.calculate_initial_position(vertex).await;
        
        let positioned = PositionedVertex {
            vertex: vertex.clone(),
            position,
            velocity: Vector2::new(0.0, 0.0),
            quantum_hue: self.calculate_quantum_hue(vertex),
            entanglement_radius: vertex.entanglement_strength() * 50.0 + 10.0,
        };

        // Add vertex
        {
            let mut vertices = self.vertices.write().await;
            vertices.insert(vertex.id, positioned);
        }

        // Add edges to parents
        {
            let mut edges = self.edges.write().await;
            for parent_id in &vertex.parents {
                let edge = self.create_quantum_edge(*parent_id, vertex.id, vertex).await;
                edges.push(edge);
            }
        }

        // Update layout with spring forces
        self.layout_engine.update_positions(&self.vertices, &self.edges).await?;

        tracing::debug!("Added vertex {} to DAG visualization", hex::encode(vertex.id));
        Ok(())
    }

    /// Calculate initial position for a new vertex
    async fn calculate_initial_position(&self, vertex: &Vertex) -> Point2<f64> {
        let vertices = self.vertices.read().await;
        
        if vertex.parents.is_empty() {
            // Genesis vertex at center
            Point2::new(
                self.canvas_size.0 as f64 / 2.0,
                self.canvas_size.1 as f64 / 2.0,
            )
        } else {
            // Position near parent centers with quantum perturbation
            let parent_positions: Vec<Point2<f64>> = vertex.parents
                .iter()
                .filter_map(|pid| vertices.get(pid).map(|pv| pv.position))
                .collect();

            if parent_positions.is_empty() {
                // Fallback to random position
                Point2::new(
                    rand::random::<f64>() * self.canvas_size.0 as f64,
                    rand::random::<f64>() * self.canvas_size.1 as f64,
                )
            } else {
                // Average of parent positions with quantum noise
                let avg = parent_positions.iter().fold(Point2::new(0.0, 0.0), |acc, p| {
                    Point2::new(acc.x + p.x, acc.y + p.y)
                }) / parent_positions.len() as f64;

                // Add quantum-derived perturbation
                let phase = vertex.quantum_phase();
                let perturbation = Vector2::new(
                    phase.sin() * 100.0,
                    phase.cos() * 100.0,
                );

                avg + perturbation
            }
        }
    }

    /// Calculate quantum hue based on vertex entropy
    fn calculate_quantum_hue(&self, vertex: &Vertex) -> f32 {
        if let Some(entropy) = vertex.quantum_entropy() {
            // Map entropy to hue [0, 360]
            let entropy_sum: u32 = entropy.iter().map(|&b| b as u32).sum();
            (entropy_sum % 360) as f32
        } else {
            // Classical gray
            0.0
        }
    }

    /// Create quantum edge with interference pattern
    async fn create_quantum_edge(
        &self,
        from: VertexId,
        to: VertexId,
        to_vertex: &Vertex,
    ) -> QuantumEdge {
        let strength = to_vertex.entanglement_strength();
        let phase_shift = to_vertex.quantum_phase();
        
        // Generate interference pattern points along edge
        let interference_pattern = self.generate_interference_pattern(
            from, to, strength, phase_shift
        ).await;

        QuantumEdge {
            from,
            to,
            strength,
            phase_shift,
            interference_pattern,
        }
    }

    /// Generate interference pattern for quantum edge
    async fn generate_interference_pattern(
        &self,
        from: VertexId,
        to: VertexId,
        strength: f64,
        phase: f64,
    ) -> Vec<Point2<f64>> {
        let vertices = self.vertices.read().await;
        
        let from_pos = vertices.get(&from).map(|v| v.position)
            .unwrap_or_else(|| Point2::new(0.0, 0.0));
        let to_pos = vertices.get(&to).map(|v| v.position)
            .unwrap_or_else(|| Point2::new(0.0, 0.0));

        let mut pattern = Vec::new();
        let num_points = (strength * 20.0 + 5.0) as usize;

        for i in 0..num_points {
            let t = i as f64 / num_points as f64;
            
            // Linear interpolation with sinusoidal perturbation
            let base_point = from_pos + (to_pos - from_pos) * t;
            
            // Quantum interference amplitude
            let amplitude = strength * 10.0 * (phase + t * 2.0 * std::f64::consts::PI).sin();
            
            // Perpendicular offset for wave pattern
            let direction = (to_pos - from_pos).normalize();
            let perpendicular = Vector2::new(-direction.y, direction.x);
            
            let interference_point = base_point + perpendicular * amplitude;
            pattern.push(interference_point);
        }

        pattern
    }

    /// Render DAG to SVG with quantum aesthetics
    pub async fn render_to_svg(&self) -> Result<String> {
        let mut document = Document::new()
            .set("width", self.canvas_size.0)
            .set("height", self.canvas_size.1)
            .set("viewBox", (0, 0, self.canvas_size.0, self.canvas_size.1));

        // Add background gradient
        document = self.add_quantum_background(document).await;

        // Render edges with interference patterns
        document = self.render_edges_to_svg(document).await?;

        // Render vertices with entanglement halos
        document = self.render_vertices_to_svg(document).await?;

        Ok(document.to_string())
    }

    /// Add quantum gradient background
    async fn add_quantum_background(&self, mut document: Document) -> Document {
        use svg::node::element::{Definitions, RadialGradient, Stop};

        let defs = Definitions::new();
        let gradient = RadialGradient::new()
            .set("id", "quantum-bg")
            .set("cx", "50%")
            .set("cy", "50%")
            .set("r", "50%")
            .add(Stop::new().set("offset", "0%").set("stop-color", "#000010"))
            .add(Stop::new().set("offset", "100%").set("stop-color", "#000040"));

        let defs = defs.add(gradient);
        document = document.add(defs);

        let background = svg::node::element::Rectangle::new()
            .set("width", "100%")
            .set("height", "100%")
            .set("fill", "url(#quantum-bg)");

        document.add(background)
    }

    /// Render edges with quantum interference patterns
    async fn render_edges_to_svg(&self, mut document: Document) -> Result<Document> {
        let edges = self.edges.read().await;
        let vertices = self.vertices.read().await;

        for edge in edges.iter() {
            if let (Some(from_vertex), Some(to_vertex)) = (
                vertices.get(&edge.from),
                vertices.get(&edge.to)
            ) {
                // Create path with interference pattern
                let mut path_data = format!(
                    "M {} {}",
                    from_vertex.position.x,
                    from_vertex.position.y
                );

                // Add interference points as smooth curve
                for point in &edge.interference_pattern {
                    path_data.push_str(&format!(" L {} {}", point.x, point.y));
                }

                path_data.push_str(&format!(
                    " L {} {}",
                    to_vertex.position.x,
                    to_vertex.position.y
                ));

                // Color based on entanglement strength
                let hue = edge.phase_shift / (2.0 * std::f64::consts::PI) * 360.0;
                let color = format!("hsl({}, 80%, 60%)", hue as u32 % 360);

                let path = svg::node::element::Path::new()
                    .set("d", path_data)
                    .set("stroke", color)
                    .set("stroke-width", edge.strength * 3.0 + 1.0)
                    .set("fill", "none")
                    .set("opacity", 0.7);

                document = document.add(path);
            }
        }

        Ok(document)
    }

    /// Render vertices with quantum entanglement halos
    async fn render_vertices_to_svg(&self, mut document: Document) -> Result<Document> {
        let vertices = self.vertices.read().await;

        for positioned_vertex in vertices.values() {
            let vertex = &positioned_vertex.vertex;
            let pos = positioned_vertex.position;

            // Entanglement halo (outer ring)
            let halo = Circle::new()
                .set("cx", pos.x)
                .set("cy", pos.y)
                .set("r", positioned_vertex.entanglement_radius)
                .set("fill", "none")
                .set("stroke", format!("hsl({}, 70%, 50%)", positioned_vertex.quantum_hue))
                .set("stroke-width", 2)
                .set("opacity", 0.3);

            // Core vertex (inner circle)
            let core_radius = 8.0 + vertex.transactions.len() as f64 * 2.0;
            let core = Circle::new()
                .set("cx", pos.x)
                .set("cy", pos.y)
                .set("r", core_radius)
                .set("fill", format!("hsl({}, 80%, 70%)", positioned_vertex.quantum_hue))
                .set("stroke", "#ffffff")
                .set("stroke-width", 1);

            // Round number label
            let label = Text::new()
                .set("x", pos.x)
                .set("y", pos.y + 4.0)
                .set("text-anchor", "middle")
                .set("font-family", "monospace")
                .set("font-size", "10")
                .set("fill", "#ffffff")
                .add(svg::node::Text::new(format!("R{}", vertex.round)));

            document = document.add(halo).add(core).add(label);
        }

        Ok(document)
    }

    /// Get current vertex count
    pub async fn vertex_count(&self) -> u64 {
        self.vertices.read().await.len() as u64
    }

    /// Export quantum coherence metrics
    pub async fn export_coherence_metrics(&self) -> QuantumCoherenceMetrics {
        let vertices = self.vertices.read().await;
        let edges = self.edges.read().await;

        let total_vertices = vertices.len();
        let total_edges = edges.len();
        
        // Calculate average entanglement strength
        let avg_entanglement = if total_edges > 0 {
            edges.iter().map(|e| e.strength).sum::<f64>() / total_edges as f64
        } else {
            0.0
        };

        // Calculate quantum coherence index
        let qci = if total_vertices > 1 {
            let mut coherence_sum = 0.0;
            let mut pairs = 0;

            for vertex in vertices.values() {
                for other in vertices.values() {
                    if vertex.vertex.id != other.vertex.id {
                        let phase_diff = (vertex.quantum_hue - other.quantum_hue).abs() as f64;
                        let coherence = (-phase_diff / 180.0 * std::f64::consts::PI).cos().abs();
                        coherence_sum += coherence;
                        pairs += 1;
                    }
                }
            }

            if pairs > 0 { coherence_sum / pairs as f64 } else { 0.0 }
        } else {
            0.0
        };

        QuantumCoherenceMetrics {
            total_vertices: total_vertices as u64,
            total_edges: total_edges as u64,
            average_entanglement_strength: avg_entanglement,
            quantum_coherence_index: qci,
        }
    }
}

impl SpringLayoutEngine {
    pub fn new() -> Self {
        Self {
            spring_constant: 0.1,
            damping: 0.9,
            repulsion_strength: 1000.0,
            attraction_strength: 0.01,
            ideal_edge_length: 150.0,
        }
    }

    /// Update vertex positions using spring-force algorithm
    pub async fn update_positions(
        &self,
        vertices: &RwLock<HashMap<VertexId, PositionedVertex>>,
        edges: &RwLock<Vec<QuantumEdge>>,
    ) -> Result<()> {
        // Run several iterations for stability
        for _ in 0..10 {
            self.apply_forces(vertices, edges).await?;
        }
        Ok(())
    }

    async fn apply_forces(
        &self,
        vertices: &RwLock<HashMap<VertexId, PositionedVertex>>,
        edges: &RwLock<Vec<QuantumEdge>>,
    ) -> Result<()> {
        let mut forces: HashMap<VertexId, Vector2<f64>> = HashMap::new();

        // Calculate repulsion forces between all vertex pairs
        {
            let vertices_read = vertices.read().await;
            
            for (id1, v1) in vertices_read.iter() {
                for (id2, v2) in vertices_read.iter() {
                    if id1 != id2 {
                        let diff = v1.position - v2.position;
                        let distance = diff.magnitude();
                        
                        if distance > 0.0 {
                            let force_magnitude = self.repulsion_strength / (distance * distance);
                            let force = diff.normalize() * force_magnitude;
                            
                            forces.entry(*id1)
                                .or_insert_with(Vector2::zeros)
                                .add_assign(force);
                        }
                    }
                }
            }
        }

        // Calculate attraction forces along edges
        {
            let vertices_read = vertices.read().await;
            let edges_read = edges.read().await;
            
            for edge in edges_read.iter() {
                if let (Some(from_vertex), Some(to_vertex)) = (
                    vertices_read.get(&edge.from),
                    vertices_read.get(&edge.to),
                ) {
                    let diff = to_vertex.position - from_vertex.position;
                    let distance = diff.magnitude();
                    
                    if distance > 0.0 {
                        let spring_force = self.spring_constant * 
                            (distance - self.ideal_edge_length) * edge.strength;
                        let force = diff.normalize() * spring_force;
                        
                        forces.entry(edge.from)
                            .or_insert_with(Vector2::zeros)
                            .add_assign(force);
                            
                        forces.entry(edge.to)
                            .or_insert_with(Vector2::zeros)
                            .add_assign(-force);
                    }
                }
            }
        }

        // Apply forces and update positions
        {
            let mut vertices_write = vertices.write().await;
            
            for (vertex_id, force) in forces {
                if let Some(vertex) = vertices_write.get_mut(&vertex_id) {
                    vertex.velocity = vertex.velocity * self.damping + force;
                    vertex.position += vertex.velocity;
                    
                    // Keep within canvas bounds
                    vertex.position.x = vertex.position.x.clamp(50.0, 1870.0);
                    vertex.position.y = vertex.position.y.clamp(50.0, 1030.0);
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantumCoherenceMetrics {
    pub total_vertices: u64,
    pub total_edges: u64,
    pub average_entanglement_strength: f64,
    pub quantum_coherence_index: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex(id: u8, round: u64) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [1; 32],
            tx_root: [id.wrapping_mul(17); 32], // Pseudo-random entropy
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_dag_renderer_creation() {
        let renderer = DAGRenderer::new();
        assert!(renderer.is_ok());
    }

    #[tokio::test]
    async fn test_add_vertex() {
        let renderer = DAGRenderer::new().unwrap();
        let vertex = create_test_vertex(42, 1);
        
        let result = renderer.add_vertex(&vertex).await;
        assert!(result.is_ok());
        
        let count = renderer.vertex_count().await;
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_quantum_hue_calculation() {
        let renderer = DAGRenderer::new().unwrap();
        let vertex = create_test_vertex(42, 1);
        
        let hue = renderer.calculate_quantum_hue(&vertex);
        assert!(hue >= 0.0 && hue < 360.0);
    }

    #[tokio::test]
    async fn test_svg_rendering() {
        let renderer = DAGRenderer::new().unwrap();
        let vertex = create_test_vertex(42, 1);
        
        renderer.add_vertex(&vertex).await.unwrap();
        let svg = renderer.render_to_svg().await.unwrap();
        
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}