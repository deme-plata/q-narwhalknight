/// Visualization system for water-robot blockchain simulation
///
/// This module provides real-time visualization of droplet movements,
/// DNA synthesis, and biological consensus processes.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::{DropletNode, MitochondriaNetwork, ElectroWettingGrid, BiologicalConsensus};

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub scale_factor: f64,          // Pixels per mm
    pub update_rate_fps: u32,
    pub show_voltage_field: bool,
    pub show_dna_chains: bool,
    pub show_tor_connections: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            window_width: 1200,
            window_height: 800,
            scale_factor: 50.0,     // 50 pixels per mm
            update_rate_fps: 30,
            show_voltage_field: true,
            show_dna_chains: true,
            show_tor_connections: false,
        }
    }
}

/// Visual representation of droplet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropletVisual {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color: [f32; 3],            // RGB color based on state
    pub energy_glow: f32,           // Glow intensity based on energy
    pub dna_trail_length: usize,    // Length of DNA synthesis trail
}

/// Visual representation of voltage field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageFieldVisual {
    pub grid_points: Vec<Vec<f32>>, // Voltage values for visualization
    pub max_voltage: f32,
    pub field_vectors: Vec<(f32, f32, f32, f32)>, // (x, y, dx, dy) for field lines
}

/// Network visualization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkVisualization {
    pub droplets: HashMap<String, DropletVisual>,
    pub voltage_field: VoltageFieldVisual,
    pub consensus_info: ConsensusVisual,
    pub simulation_time: f64,
    pub network_stats: NetworkStats,
}

/// Consensus visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVisual {
    pub leader_droplet: Option<String>,
    pub total_dna_mass: f32,
    pub consensus_confidence: f32,
    pub active_validators: usize,
}

/// Network statistics for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_droplets: usize,
    pub average_energy: f32,
    pub total_dna_synthesis: f32,
    pub network_health: f32,
}

/// Create visual representation from network state
pub fn create_network_visualization(
    network: &MitochondriaNetwork,
    config: &VisualizationConfig
) -> Result<NetworkVisualization> {
    let mut droplet_visuals = HashMap::new();
    
    // Create droplet visuals
    for (id, droplet) in &network.droplets {
        let visual = create_droplet_visual(droplet, config);
        droplet_visuals.insert(id.clone(), visual);
    }
    
    // Create voltage field visualization
    let voltage_field = create_voltage_field_visual(&network.electro_wetting_grid, config);
    
    // Create consensus visualization
    let consensus_info = create_consensus_visual(&network.consensus_state, &network.droplets);
    
    // Calculate network statistics
    let network_stats = calculate_network_stats(&network.droplets);
    
    Ok(NetworkVisualization {
        droplets: droplet_visuals,
        voltage_field,
        consensus_info,
        simulation_time: network.simulation_time.timestamp_millis() as f64 / 1000.0,
        network_stats,
    })
}

/// Create visual representation of a droplet
fn create_droplet_visual(droplet: &DropletNode, config: &VisualizationConfig) -> DropletVisual {
    // Convert position to screen coordinates
    let x = (droplet.position.x * config.scale_factor) as f32;
    let y = (droplet.position.y * config.scale_factor) as f32;
    
    // Radius based on droplet size
    let radius = ((droplet.size_nanoliters / 10.0).sqrt() * 5.0) as f32;
    
    // Color based on DNA chain length and energy
    let color = calculate_droplet_color(droplet);
    
    // Glow intensity based on energy level
    let energy_glow = (droplet.energy_level as f32).powf(2.0);
    
    // DNA trail length based on synthesis history
    let dna_trail_length = droplet.dna_data.synthesis_history.len().min(20);
    
    DropletVisual {
        x,
        y,
        radius,
        color,
        energy_glow,
        dna_trail_length,
    }
}

/// Calculate droplet color based on state
fn calculate_droplet_color(droplet: &DropletNode) -> [f32; 3] {
    let dna_factor = (droplet.dna_data.total_mass_picograms / 50.0).min(1.0) as f32;
    let energy_factor = droplet.energy_level as f32;
    
    // Base color is blue-green (biological)
    let red = 0.2 + dna_factor * 0.3;
    let green = 0.4 + energy_factor * 0.4;
    let blue = 0.6 + dna_factor * 0.2;
    
    [red, green, blue]
}

/// Create voltage field visualization
fn create_voltage_field_visual(
    grid: &ElectroWettingGrid,
    config: &VisualizationConfig
) -> VoltageFieldVisual {
    let grid_size = grid.voltage_matrix.len();
    let mut max_voltage = 0.0f32;
    let mut grid_points = Vec::with_capacity(grid_size);
    
    // Convert voltage matrix to visual format
    for row in &grid.voltage_matrix {
        let mut visual_row = Vec::with_capacity(row.len());
        for &voltage in row {
            let v = voltage as f32;
            visual_row.push(v);
            max_voltage = max_voltage.max(v.abs());
        }
        grid_points.push(visual_row);
    }
    
    // Create field vectors for visualization
    let field_vectors = create_voltage_field_vectors(grid, config);
    
    VoltageFieldVisual {
        grid_points,
        max_voltage,
        field_vectors,
    }
}

/// Create field vectors for voltage visualization
fn create_voltage_field_vectors(
    grid: &ElectroWettingGrid,
    config: &VisualizationConfig
) -> Vec<(f32, f32, f32, f32)> {
    let mut vectors = Vec::new();
    let grid_size = grid.voltage_matrix.len();
    let spacing = (grid.pad_spacing_um * config.scale_factor / 1000.0) as f32; // Convert to screen units
    
    for x in 1..grid_size-1 {
        for y in 1..grid.voltage_matrix[0].len()-1 {
            // Calculate gradient
            let grad_x = (grid.voltage_matrix[x+1][y] - grid.voltage_matrix[x-1][y]) / 2.0;
            let grad_y = (grid.voltage_matrix[x][y+1] - grid.voltage_matrix[x][y-1]) / 2.0;
            
            if grad_x.abs() > 0.1 || grad_y.abs() > 0.1 {
                let start_x = x as f32 * spacing;
                let start_y = y as f32 * spacing;
                let end_x = start_x + grad_x as f32 * spacing * 0.5;
                let end_y = start_y + grad_y as f32 * spacing * 0.5;
                
                vectors.push((start_x, start_y, end_x, end_y));
            }
        }
    }
    
    vectors
}

/// Create consensus visualization data
fn create_consensus_visual(
    consensus: &BiologicalConsensus,
    droplets: &HashMap<String, DropletNode>
) -> ConsensusVisual {
    let active_validators = droplets
        .values()
        .filter(|d| d.dna_data.total_mass_picograms > 5.0 && d.energy_level > 0.3)
        .count();
    
    ConsensusVisual {
        leader_droplet: Some(consensus.heaviest_swarm_leader.clone()),
        total_dna_mass: consensus.total_network_dna_mass as f32,
        consensus_confidence: consensus.consensus_confidence as f32,
        active_validators,
    }
}

/// Calculate network statistics for display
fn calculate_network_stats(droplets: &HashMap<String, DropletNode>) -> NetworkStats {
    if droplets.is_empty() {
        return NetworkStats {
            total_droplets: 0,
            average_energy: 0.0,
            total_dna_synthesis: 0.0,
            network_health: 0.0,
        };
    }
    
    let total_energy: f64 = droplets.values().map(|d| d.energy_level).sum();
    let total_dna: f64 = droplets.values().map(|d| d.dna_data.total_mass_picograms).sum();
    
    let average_energy = (total_energy / droplets.len() as f64) as f32;
    let network_health = calculate_network_health(droplets);
    
    NetworkStats {
        total_droplets: droplets.len(),
        average_energy,
        total_dna_synthesis: total_dna as f32,
        network_health,
    }
}

/// Calculate overall network health score
fn calculate_network_health(droplets: &HashMap<String, DropletNode>) -> f32 {
    if droplets.is_empty() {
        return 0.0;
    }
    
    let healthy_droplets = droplets
        .values()
        .filter(|d| d.energy_level > 0.5 && d.dna_data.total_mass_picograms > 1.0)
        .count();
    
    (healthy_droplets as f32 / droplets.len() as f32) * 100.0
}

/// Generate HTML visualization report
pub fn generate_html_report(visualization: &NetworkVisualization) -> String {
    format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Project Mitochondria v2 - Biological Blockchain Visualization</title>
    <style>
        body {{ font-family: monospace; background: #001122; color: #00ff88; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .stat-box {{ background: #002244; padding: 15px; border-radius: 8px; border: 1px solid #004466; }}
        .droplet-list {{ max-height: 400px; overflow-y: auto; background: #001122; border: 1px solid #004466; padding: 10px; }}
        .droplet-item {{ margin-bottom: 10px; padding: 8px; background: #002244; border-radius: 4px; }}
        .consensus-info {{ background: #003344; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Project Mitochondria v2 - Water-Robot Blockchain</h1>
        <p>Biological consensus simulation at time: {:.2}s</p>
        
        <div class="consensus-info">
            <h2>👑 Biological Consensus State</h2>
            <p><strong>Leader Droplet:</strong> {}</p>
            <p><strong>Total DNA Mass:</strong> {:.2} pg</p>
            <p><strong>Consensus Confidence:</strong> {:.1}%</p>
            <p><strong>Active Validators:</strong> {}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <h3>💧 Total Droplets</h3>
                <p style="font-size: 24px;">{}</p>
            </div>
            <div class="stat-box">
                <h3>⚡ Average Energy</h3>
                <p style="font-size: 24px;">{:.1}%</p>
            </div>
            <div class="stat-box">
                <h3>🧬 Total DNA</h3>
                <p style="font-size: 24px;">{:.2} pg</p>
            </div>
            <div class="stat-box">
                <h3>❤️ Network Health</h3>
                <p style="font-size: 24px;">{:.1}%</p>
            </div>
        </div>
        
        <div class="droplet-list">
            <h3>Active Droplets</h3>
            {}
        </div>
    </div>
</body>
</html>
"#,
        visualization.simulation_time,
        visualization.consensus_info.leader_droplet.as_deref().unwrap_or("None"),
        visualization.consensus_info.total_dna_mass,
        visualization.consensus_info.consensus_confidence * 100.0,
        visualization.consensus_info.active_validators,
        visualization.network_stats.total_droplets,
        visualization.network_stats.average_energy * 100.0,
        visualization.network_stats.total_dna_synthesis,
        visualization.network_stats.network_health,
        generate_droplet_list_html(&visualization.droplets)
    )
}

/// Generate HTML for droplet list
fn generate_droplet_list_html(droplets: &HashMap<String, DropletVisual>) -> String {
    let mut html = String::new();
    
    for (id, visual) in droplets {
        html.push_str(&format!(
            r#"<div class="droplet-item">
                <strong>{}</strong> - Position: ({:.1}, {:.1}) | 
                Energy: {:.1}% | DNA Trail: {} blocks
            </div>"#,
            id,
            visual.x / 50.0, // Convert back to mm
            visual.y / 50.0,
            visual.energy_glow * 100.0,
            visual.dna_trail_length
        ));
    }
    
    html
}

/// Save visualization to file
pub async fn save_visualization_snapshot(
    visualization: &NetworkVisualization,
    filename: &str
) -> Result<()> {
    let html_content = generate_html_report(visualization);
    tokio::fs::write(filename, html_content).await?;
    
    info!("📊 Saved visualization snapshot to: {}", filename);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Position2D, DNABlockchain, DNASynthesisEvent};
    use chrono::Utc;

    fn create_test_droplet(id: &str, x: f64, y: f64) -> DropletNode {
        DropletNode {
            droplet_id: id.to_string(),
            position: Position2D { x, y, velocity_x: 0.0, velocity_y: 0.0 },
            dna_data: DNABlockchain {
                chain_length: 2,
                genesis_hash: "test".to_string(),
                latest_block_hash: "test".to_string(),
                total_mass_picograms: 10.0,
                synthesis_history: vec![
                    DNASynthesisEvent {
                        block_height: 0,
                        sequence_added: "ATGC".to_string(),
                        synthesis_time_ms: 100,
                        energy_cost: 0.5,
                        synthesized_at: Utc::now(),
                    }
                ],
            },
            energy_level: 0.8,
            size_nanoliters: 15.0,
        }
    }

    #[test]
    fn test_droplet_visual_creation() {
        let droplet = create_test_droplet("test", 2.0, 3.0);
        let config = VisualizationConfig::default();
        
        let visual = create_droplet_visual(&droplet, &config);
        
        assert_eq!(visual.x, 100.0); // 2.0 * 50.0 scale
        assert_eq!(visual.y, 150.0); // 3.0 * 50.0 scale
        assert!(visual.radius > 0.0);
        assert!(visual.energy_glow > 0.0);
    }

    #[test]
    fn test_network_stats_calculation() {
        let mut droplets = HashMap::new();
        droplets.insert("droplet1".to_string(), create_test_droplet("droplet1", 1.0, 1.0));
        droplets.insert("droplet2".to_string(), create_test_droplet("droplet2", 2.0, 2.0));
        
        let stats = calculate_network_stats(&droplets);
        
        assert_eq!(stats.total_droplets, 2);
        assert!(stats.average_energy > 0.0);
        assert!(stats.total_dna_synthesis > 0.0);
        assert!(stats.network_health > 0.0);
    }

    #[test]
    fn test_html_report_generation() {
        let visualization = NetworkVisualization {
            droplets: HashMap::new(),
            voltage_field: VoltageFieldVisual {
                grid_points: vec![vec![0.0; 10]; 10],
                max_voltage: 100.0,
                field_vectors: vec![],
            },
            consensus_info: ConsensusVisual {
                leader_droplet: Some("droplet1".to_string()),
                total_dna_mass: 50.0,
                consensus_confidence: 0.95,
                active_validators: 3,
            },
            simulation_time: 123.45,
            network_stats: NetworkStats {
                total_droplets: 5,
                average_energy: 0.8,
                total_dna_synthesis: 50.0,
                network_health: 80.0,
            },
        };
        
        let html = generate_html_report(&visualization);
        
        assert!(html.contains("Project Mitochondria v2"));
        assert!(html.contains("droplet1"));
        assert!(html.contains("50.0 pg"));
        assert!(html.contains("95.0%"));
    }
}