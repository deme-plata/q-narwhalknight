/// Q-NarwhalKnight Quantum Visualization Engine
/// Creates beautiful real-time visualizations of quantum consensus

use q_types::*;
use anyhow::Result;
use std::collections::HashMap;
use tokio::sync::RwLock;

pub mod dag_renderer;
pub mod quantum_state;
pub mod qkd_waterfall;
pub mod stark_fractals;
pub mod terminal_ui;
pub mod web_interface;
pub mod moire_patterns;

pub use dag_renderer::DAGRenderer;
pub use quantum_state::{QuantumStateVisualizer, BeaconState};
pub use qkd_waterfall::QKDWaterfall;
pub use stark_fractals::STARKFractalGenerator;
pub use terminal_ui::TerminalInterface;
pub use web_interface::WebVisualizer;
pub use moire_patterns::MoirePatternGenerator;

/// Main visualization engine coordinating all visual components
pub struct QuantumVisualizer {
    pub dag_renderer: DAGRenderer,
    pub quantum_state: QuantumStateVisualizer,
    pub qkd_waterfall: QKDWaterfall,
    pub stark_fractals: STARKFractalGenerator,
    pub moire_generator: MoirePatternGenerator,
    pub terminal_ui: TerminalInterface,
    pub web_interface: WebVisualizer,
}

impl QuantumVisualizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            dag_renderer: DAGRenderer::new()?,
            quantum_state: QuantumStateVisualizer::new()?,
            qkd_waterfall: QKDWaterfall::new()?,
            stark_fractals: STARKFractalGenerator::new()?,
            moire_generator: MoirePatternGenerator::new()?,
            terminal_ui: TerminalInterface::new()?,
            web_interface: WebVisualizer::new()?,
        })
    }

    /// Start the visualization engine with all components
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("🎨 Starting Q-NarwhalKnight Quantum Visualizer");

        // Start all visualization components concurrently
        let mut tasks = vec![];

        // Terminal UI
        let terminal_handle = {
            let mut terminal = self.terminal_ui.clone();
            tokio::spawn(async move {
                terminal.run().await
            })
        };
        tasks.push(terminal_handle);

        // Web interface
        let web_handle = {
            let mut web = self.web_interface.clone();
            tokio::spawn(async move {
                web.start_server().await
            })
        };
        tasks.push(web_handle);

        // QKD waterfall animation
        let qkd_handle = {
            let mut qkd = self.qkd_waterfall.clone();
            tokio::spawn(async move {
                qkd.start_animation().await
            })
        };
        tasks.push(qkd_handle);

        tracing::info!("✅ All visualization components started");

        // Wait for any component to exit
        futures::future::try_join_all(tasks).await?;

        Ok(())
    }

    /// Process a new vertex and update all visualizations
    pub async fn process_vertex(&mut self, vertex: &Vertex) -> Result<()> {
        // Update DAG visualization
        self.dag_renderer.add_vertex(vertex).await?;

        // Generate Moiré patterns if quantum entropy is present
        if let Some(entropy) = vertex.quantum_entropy() {
            self.moire_generator.update_with_entropy(vertex.id, entropy).await?;
        }

        // Update terminal display
        self.terminal_ui.update_vertex_count().await?;

        Ok(())
    }

    /// Process quantum beacon state update
    pub async fn process_beacon_update(&mut self, beacon: &BeaconState) -> Result<()> {
        // Update quantum state rainbow-box visualization
        self.quantum_state.update_beacon(beacon).await?;

        // Update terminal quantum state display
        self.terminal_ui.update_quantum_state(beacon).await?;

        Ok(())
    }

    /// Process QKD photon detection event
    pub async fn process_qkd_event(&mut self, photon_count: u64, qber: f64) -> Result<()> {
        // Update photon waterfall
        self.qkd_waterfall.add_photons(photon_count, qber).await?;

        // Update terminal QKD status
        self.terminal_ui.update_qkd_status(photon_count, qber).await?;

        Ok(())
    }

    /// Process STARK proof generation
    pub async fn process_stark_proof(&mut self, proof_data: &[u8]) -> Result<String> {
        // Generate fractal SVG from proof
        let svg_path = self.stark_fractals.generate_fractal(proof_data).await?;

        // Update web interface with new fractal
        self.web_interface.update_stark_fractal(&svg_path).await?;

        Ok(svg_path)
    }

    /// Get current visualization metrics
    pub async fn get_metrics(&self) -> VisualizationMetrics {
        VisualizationMetrics {
            total_vertices_rendered: self.dag_renderer.vertex_count().await,
            moire_patterns_generated: self.moire_generator.pattern_count().await,
            qkd_photons_processed: self.qkd_waterfall.total_photons().await,
            stark_fractals_generated: self.stark_fractals.fractal_count().await,
            quantum_coherence_index: self.quantum_state.coherence_index().await,
            web_clients_connected: self.web_interface.client_count().await,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VisualizationMetrics {
    pub total_vertices_rendered: u64,
    pub moire_patterns_generated: u64,
    pub qkd_photons_processed: u64,
    pub stark_fractals_generated: u64,
    pub quantum_coherence_index: f64,
    pub web_clients_connected: u64,
}

/// Color palette for quantum visualizations
#[derive(Debug, Clone)]
pub struct QuantumColorPalette {
    pub entangled_blue: palette::Srgb<u8>,
    pub superposition_green: palette::Srgb<u8>,
    pub decoherence_red: palette::Srgb<u8>,
    pub classical_gray: palette::Srgb<u8>,
    pub quantum_purple: palette::Srgb<u8>,
}

impl Default for QuantumColorPalette {
    fn default() -> Self {
        use palette::Srgb;
        Self {
            entangled_blue: Srgb::new(0, 119, 190),
            superposition_green: Srgb::new(0, 153, 76),
            decoherence_red: Srgb::new(220, 20, 60),
            classical_gray: Srgb::new(128, 128, 128),
            quantum_purple: Srgb::new(102, 45, 145),
        }
    }
}

/// Extension trait to add quantum properties to vertices
pub trait QuantumVertexExt {
    fn quantum_entropy(&self) -> Option<[u8; 32]>;
    fn quantum_phase(&self) -> f64;
    fn entanglement_strength(&self) -> f64;
    fn coherence_time(&self) -> f64;
}

impl QuantumVertexExt for Vertex {
    fn quantum_entropy(&self) -> Option<[u8; 32]> {
        // For Phase 0, extract from transaction data
        // TODO: Use actual quantum entropy field from Phase 2+
        if self.transactions.is_empty() {
            None
        } else {
            Some(self.tx_root)
        }
    }

    fn quantum_phase(&self) -> f64 {
        // Calculate quantum phase from entropy
        if let Some(entropy) = self.quantum_entropy() {
            let sum: u64 = entropy.iter().map(|&b| b as u64).sum();
            (sum as f64 / 255.0 / 32.0) * 2.0 * std::f64::consts::PI
        } else {
            0.0
        }
    }

    fn entanglement_strength(&self) -> f64 {
        // Calculate entanglement strength from parent correlations
        if self.parents.is_empty() {
            0.0
        } else {
            // Simple correlation measure
            let parent_hash_sum: u64 = self.parents.iter()
                .flat_map(|p| p.iter())
                .map(|&b| b as u64)
                .sum();
            
            let self_hash_sum: u64 = self.id.iter().map(|&b| b as u64).sum();
            
            (parent_hash_sum as f64 * self_hash_sum as f64).sqrt() / (u64::MAX as f64)
        }
    }

    fn coherence_time(&self) -> f64 {
        // Estimate quantum coherence time from entropy stability
        // Higher entropy = longer coherence time
        if let Some(entropy) = self.quantum_entropy() {
            let entropy_measure = shannon_entropy(&entropy);
            entropy_measure * 100.0 // Scale to milliseconds
        } else {
            0.0
        }
    }
}

/// Calculate Shannon entropy of byte array
fn shannon_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for count in counts.iter() {
        if *count > 0 {
            let p = *count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy / 8.0 // Normalize to [0, 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex() -> Vertex {
        Vertex {
            id: [42u8; 32],
            round: 1,
            author: [1u8; 32],
            tx_root: [0xAB; 32], // High entropy pattern
            parents: vec![[0u8; 32], [255u8; 32]],
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_quantum_properties() {
        let vertex = create_test_vertex();
        
        let entropy = vertex.quantum_entropy();
        assert!(entropy.is_some());
        
        let phase = vertex.quantum_phase();
        assert!(phase >= 0.0 && phase <= 2.0 * std::f64::consts::PI);
        
        let entanglement = vertex.entanglement_strength();
        assert!(entanglement >= 0.0 && entanglement <= 1.0);
        
        let coherence = vertex.coherence_time();
        assert!(coherence >= 0.0);
    }

    #[test]
    fn test_shannon_entropy() {
        let uniform_data = [0xAB; 32]; // Low entropy
        let uniform_entropy = shannon_entropy(&uniform_data);
        assert!(uniform_entropy < 0.1);
        
        let random_data: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]; // Higher entropy
        let random_entropy = shannon_entropy(&random_data);
        assert!(random_entropy > 0.8);
    }

    #[tokio::test]
    async fn test_visualizer_creation() {
        let visualizer = QuantumVisualizer::new();
        assert!(visualizer.is_ok());
    }
}