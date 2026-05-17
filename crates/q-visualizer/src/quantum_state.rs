/// Quantum state visualization using rainbow-box technique
/// Renders multi-qubit beacon states with superposition and entanglement

use anyhow::Result;
use ndarray::{Array2, Array1};
use num_complex::Complex64;
use palette::{Hsl, Srgb, FromColor};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Quantum beacon state for consensus rounds
#[derive(Debug, Clone, serde::Serialize)]
pub struct BeaconState {
    pub round: u64,
    pub qubits: Vec<QubitState>,
    pub entanglement_matrix: Array2<Complex64>,
    pub phase_correlations: HashMap<usize, f64>,
    pub coherence_time: f64,
    pub measurement_probability: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QubitState {
    pub id: usize,
    pub alpha: Complex64,  // |0⟩ amplitude
    pub beta: Complex64,   // |1⟩ amplitude
    pub phase: f64,
    pub entangled_with: Vec<usize>,
}

/// Rainbow-box quantum state visualizer
#[derive(Clone)]
pub struct QuantumStateVisualizer {
    current_beacon: RwLock<Option<BeaconState>>,
    state_history: RwLock<Vec<BeaconState>>,
    canvas_size: (u32, u32),
    box_dimensions: (u32, u32),
}

impl QuantumStateVisualizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_beacon: RwLock::new(None),
            state_history: RwLock::new(Vec::new()),
            canvas_size: (800, 600),
            box_dimensions: (100, 80),
        })
    }

    /// Update with new beacon state
    pub async fn update_beacon(&self, beacon: &BeaconState) -> Result<()> {
        // Store current beacon
        {
            let mut current = self.current_beacon.write().await;
            *current = Some(beacon.clone());
        }

        // Add to history (keep last 100 states)
        {
            let mut history = self.state_history.write().await;
            history.push(beacon.clone());
            if history.len() > 100 {
                history.remove(0);
            }
        }

        tracing::debug!("Updated quantum beacon state for round {}", beacon.round);
        Ok(())
    }

    /// Generate rainbow-box visualization as SVG
    pub async fn render_rainbow_box(&self) -> Result<String> {
        let beacon = {
            let current = self.current_beacon.read().await;
            match current.as_ref() {
                Some(b) => b.clone(),
                None => return Ok(self.render_empty_state()),
            }
        };

        let mut svg = format!(
            r#"<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            self.canvas_size.0, self.canvas_size.1,
            self.canvas_size.0, self.canvas_size.1
        );

        // Add title
        svg.push_str(&format!(
            r#"<text x="10" y="30" font-family="monospace" font-size="16" fill="white">Round {} Beacon State</text>"#,
            beacon.round
        ));

        // Calculate grid layout
        let qubits_per_row = (self.canvas_size.0 / self.box_dimensions.0).max(1);
        let rows = (beacon.qubits.len() as u32 + qubits_per_row - 1) / qubits_per_row;

        for (i, qubit) in beacon.qubits.iter().enumerate() {
            let col = i as u32 % qubits_per_row;
            let row = i as u32 / qubits_per_row;
            
            let x = col * self.box_dimensions.0 + 50;
            let y = row * self.box_dimensions.1 + 80;

            svg.push_str(&self.render_qubit_box(qubit, x, y, &beacon)?);
        }

        // Add entanglement connections
        svg.push_str(&self.render_entanglement_lines(&beacon)?);

        // Add coherence metrics
        svg.push_str(&self.render_coherence_metrics(&beacon)?);

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Render individual qubit rainbow box
    fn render_qubit_box(
        &self,
        qubit: &QubitState,
        x: u32,
        y: u32,
        beacon: &BeaconState,
    ) -> Result<String> {
        let width = self.box_dimensions.0;
        let height = self.box_dimensions.1;

        // Calculate probabilities
        let prob_0 = qubit.alpha.norm_sqr();
        let prob_1 = qubit.beta.norm_sqr();

        // Map phase to hue (0-360 degrees)
        let hue = (qubit.phase / (2.0 * std::f64::consts::PI) * 360.0) % 360.0;

        // Create gradient based on superposition
        let mut box_svg = String::new();

        // Background gradient from |0⟩ to |1⟩ probabilities
        let gradient_id = format!("gradient_qubit_{}", qubit.id);
        box_svg.push_str(&format!(
            r#"<defs>
                <linearGradient id="{}" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="hsl({}, 70%, {}%)" />
                    <stop offset="100%" stop-color="hsl({}, 70%, {}%)" />
                </linearGradient>
            </defs>"#,
            gradient_id,
            hue, (prob_0 * 80.0) as u32 + 20,
            (hue + 180.0) % 360.0, (prob_1 * 80.0) as u32 + 20
        ));

        // Main box
        box_svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" 
               fill="url(#{})" stroke="white" stroke-width="2" rx="5" />"#,
            x, y, width, height, gradient_id
        ));

        // Amplitude visualization (height bars)
        let bar_width = width / 3;
        let bar_0_height = (prob_0 * height as f64 * 0.8) as u32;
        let bar_1_height = (prob_1 * height as f64 * 0.8) as u32;

        // |0⟩ amplitude bar
        box_svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" 
               fill="rgba(255,255,255,0.7)" stroke="white" />"#,
            x + 10, y + height - bar_0_height - 10,
            bar_width, bar_0_height
        ));

        // |1⟩ amplitude bar  
        box_svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" 
               fill="rgba(255,255,255,0.7)" stroke="white" />"#,
            x + width - bar_width - 10, y + height - bar_1_height - 10,
            bar_width, bar_1_height
        ));

        // Labels
        box_svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="monospace" font-size="10" fill="white" text-anchor="middle">|0⟩</text>"#,
            x + 10 + bar_width/2, y + height - 5
        ));

        box_svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="monospace" font-size="10" fill="white" text-anchor="middle">|1⟩</text>"#,
            x + width - bar_width/2 - 10, y + height - 5
        ));

        // Qubit ID and probability values
        box_svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="monospace" font-size="12" fill="white" text-anchor="middle">Q{}</text>"#,
            x + width/2, y + 20, qubit.id
        ));

        box_svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="monospace" font-size="8" fill="white" text-anchor="middle">{:.3}</text>"#,
            x + 10 + bar_width/2, y + 35, prob_0
        ));

        box_svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="monospace" font-size="8" fill="white" text-anchor="middle">{:.3}</text>"#,
            x + width - bar_width/2 - 10, y + 35, prob_1
        ));

        // Entanglement indicator
        if !qubit.entangled_with.is_empty() {
            box_svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="8" fill="rgba(255,20,60,0.8)" stroke="white" stroke-width="1" />"#,
                x + width - 15, y + 15
            ));
            
            box_svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-family="monospace" font-size="8" fill="white" text-anchor="middle">{}</text>"#,
                x + width - 15, y + 19, qubit.entangled_with.len()
            ));
        }

        Ok(box_svg)
    }

    /// Render entanglement connection lines
    fn render_entanglement_lines(&self, beacon: &BeaconState) -> Result<String> {
        let mut lines_svg = String::new();
        
        let qubits_per_row = (self.canvas_size.0 / self.box_dimensions.0).max(1);
        
        for qubit in &beacon.qubits {
            for &entangled_id in &qubit.entangled_with {
                if entangled_id > qubit.id {  // Draw each line only once
                    // Calculate positions
                    let (x1, y1) = self.calculate_qubit_position(qubit.id, qubits_per_row);
                    let (x2, y2) = self.calculate_qubit_position(entangled_id, qubits_per_row);
                    
                    // Get entanglement strength from correlation matrix
                    let strength = beacon.entanglement_matrix
                        .get((qubit.id, entangled_id))
                        .map(|c| c.norm())
                        .unwrap_or(0.0);
                    
                    // Color based on entanglement strength
                    let color = format!("rgba(220,20,60,{})", strength * 0.8 + 0.2);
                    
                    lines_svg.push_str(&format!(
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" 
                           stroke="{}" stroke-width="{}" stroke-dasharray="5,5" />"#,
                        x1, y1, x2, y2, color, (strength * 3.0 + 1.0) as u32
                    ));
                    
                    // Add midpoint indicator
                    let mid_x = (x1 + x2) / 2;
                    let mid_y = (y1 + y2) / 2;
                    
                    lines_svg.push_str(&format!(
                        r#"<circle cx="{}" cy="{}" r="3" fill="{}" />"#,
                        mid_x, mid_y, color
                    ));
                }
            }
        }
        
        Ok(lines_svg)
    }

    /// Calculate qubit box center position
    fn calculate_qubit_position(&self, qubit_id: usize, qubits_per_row: u32) -> (u32, u32) {
        let col = qubit_id as u32 % qubits_per_row;
        let row = qubit_id as u32 / qubits_per_row;
        
        let x = col * self.box_dimensions.0 + 50 + self.box_dimensions.0 / 2;
        let y = row * self.box_dimensions.1 + 80 + self.box_dimensions.1 / 2;
        
        (x, y)
    }

    /// Render coherence and phase metrics
    fn render_coherence_metrics(&self, beacon: &BeaconState) -> Result<String> {
        let mut metrics_svg = String::new();
        
        let metrics_y = self.canvas_size.1 - 100;
        
        metrics_svg.push_str(&format!(
            r#"<text x="10" y="{}" font-family="monospace" font-size="12" fill="white">Coherence Time: {:.2}ms</text>"#,
            metrics_y, beacon.coherence_time
        ));
        
        metrics_svg.push_str(&format!(
            r#"<text x="10" y="{}" font-family="monospace" font-size="12" fill="white">Measurement Probability: {:.4}</text>"#,
            metrics_y + 20, beacon.measurement_probability
        ));
        
        // Calculate overall entanglement measure
        let total_entanglement: f64 = beacon.qubits.iter()
            .map(|q| q.entangled_with.len() as f64)
            .sum::<f64>() / beacon.qubits.len() as f64 / 2.0;  // Avoid double counting
        
        metrics_svg.push_str(&format!(
            r#"<text x="10" y="{}" font-family="monospace" font-size="12" fill="white">Avg Entanglement: {:.2}</text>"#,
            metrics_y + 40, total_entanglement
        ));

        // Phase correlation indicator
        let avg_phase_correlation = beacon.phase_correlations.values().sum::<f64>() 
            / beacon.phase_correlations.len().max(1) as f64;
        
        let correlation_color = if avg_phase_correlation > 0.7 {
            "lime"
        } else if avg_phase_correlation > 0.3 {
            "yellow"
        } else {
            "red"
        };
        
        metrics_svg.push_str(&format!(
            r#"<text x="10" y="{}" font-family="monospace" font-size="12" fill="{}">Phase Coherence: {:.3}</text>"#,
            metrics_y + 60, correlation_color, avg_phase_correlation
        ));
        
        Ok(metrics_svg)
    }

    /// Render empty state placeholder
    fn render_empty_state(&self) -> String {
        format!(
            r#"<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="#000040"/>
                <text x="{}" y="{}" font-family="monospace" font-size="16" fill="white" text-anchor="middle">
                    No Quantum State Available
                </text>
                <text x="{}" y="{}" font-family="monospace" font-size="12" fill="gray" text-anchor="middle">
                    Waiting for beacon update...
                </text>
            </svg>"#,
            self.canvas_size.0, self.canvas_size.1,
            self.canvas_size.0, self.canvas_size.1,
            self.canvas_size.0 / 2, self.canvas_size.1 / 2,
            self.canvas_size.0 / 2, self.canvas_size.1 / 2 + 30
        )
    }

    /// Get current quantum coherence index
    pub async fn coherence_index(&self) -> f64 {
        let beacon = self.current_beacon.read().await;
        match beacon.as_ref() {
            Some(b) => {
                // Calculate quantum coherence index from state overlaps
                let mut coherence_sum = 0.0;
                let mut pairs = 0;

                for i in 0..b.qubits.len() {
                    for j in i+1..b.qubits.len() {
                        let qubit_i = &b.qubits[i];
                        let qubit_j = &b.qubits[j];
                        
                        // Calculate state overlap
                        let overlap = (qubit_i.alpha.conj() * qubit_j.alpha + 
                                     qubit_i.beta.conj() * qubit_j.beta).norm();
                        
                        coherence_sum += overlap;
                        pairs += 1;
                    }
                }

                if pairs > 0 {
                    coherence_sum / pairs as f64
                } else {
                    0.0
                }
            },
            None => 0.0
        }
    }

    /// Generate test beacon state for visualization testing
    pub fn generate_test_beacon(round: u64, num_qubits: usize) -> BeaconState {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut qubits = Vec::with_capacity(num_qubits);
        
        for i in 0..num_qubits {
            // Generate random quantum state
            let theta = rng.gen::<f64>() * std::f64::consts::PI;
            let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            
            let alpha = Complex64::new(theta.cos(), 0.0);
            let beta = Complex64::new(theta.sin() * phi.cos(), theta.sin() * phi.sin());
            
            // Random entanglement
            let mut entangled_with = Vec::new();
            if rng.gen::<f64>() > 0.5 && i > 0 {
                entangled_with.push(rng.gen_range(0..i));
            }
            
            qubits.push(QubitState {
                id: i,
                alpha,
                beta,
                phase: phi,
                entangled_with,
            });
        }
        
        // Generate entanglement matrix
        let mut entanglement_matrix = Array2::zeros((num_qubits, num_qubits));
        for qubit in &qubits {
            for &entangled_id in &qubit.entangled_with {
                let strength = rng.gen::<f64>();
                entanglement_matrix[[qubit.id, entangled_id]] = Complex64::new(strength, 0.0);
                entanglement_matrix[[entangled_id, qubit.id]] = Complex64::new(strength, 0.0);
            }
        }
        
        // Generate phase correlations
        let mut phase_correlations = HashMap::new();
        for i in 0..num_qubits {
            phase_correlations.insert(i, rng.gen::<f64>());
        }
        
        BeaconState {
            round,
            qubits,
            entanglement_matrix,
            phase_correlations,
            coherence_time: rng.gen::<f64>() * 100.0 + 10.0,
            measurement_probability: rng.gen::<f64>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_visualizer_creation() {
        let visualizer = QuantumStateVisualizer::new().unwrap();
        let coherence = visualizer.coherence_index().await;
        assert_eq!(coherence, 0.0); // No beacon state yet
    }

    #[tokio::test]
    async fn test_beacon_update() {
        let visualizer = QuantumStateVisualizer::new().unwrap();
        let beacon = QuantumStateVisualizer::generate_test_beacon(42, 3);
        
        visualizer.update_beacon(&beacon).await.unwrap();
        
        let coherence = visualizer.coherence_index().await;
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[tokio::test]
    async fn test_svg_rendering() {
        let visualizer = QuantumStateVisualizer::new().unwrap();
        let beacon = QuantumStateVisualizer::generate_test_beacon(42, 4);
        
        visualizer.update_beacon(&beacon).await.unwrap();
        
        let svg = visualizer.render_rainbow_box().await.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Rainbow"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_beacon_generation() {
        let beacon = QuantumStateVisualizer::generate_test_beacon(100, 5);
        
        assert_eq!(beacon.round, 100);
        assert_eq!(beacon.qubits.len(), 5);
        
        // Verify quantum state normalization
        for qubit in &beacon.qubits {
            let norm_squared = qubit.alpha.norm_sqr() + qubit.beta.norm_sqr();
            assert!((norm_squared - 1.0).abs() < 0.1); // Allow some floating point error
        }
    }

    #[test]
    fn test_qubit_probabilities() {
        let beacon = QuantumStateVisualizer::generate_test_beacon(1, 1);
        let qubit = &beacon.qubits[0];
        
        let prob_0 = qubit.alpha.norm_sqr();
        let prob_1 = qubit.beta.norm_sqr();
        let total_prob = prob_0 + prob_1;
        
        assert!((total_prob - 1.0).abs() < 1e-10);
        assert!(prob_0 >= 0.0 && prob_0 <= 1.0);
        assert!(prob_1 >= 0.0 && prob_1 <= 1.0);
    }
}