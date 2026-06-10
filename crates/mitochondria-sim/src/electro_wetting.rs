/// Electro-wetting droplet control system
///
/// This module manages the electro-wetting grid that controls droplet movement
/// and positioning for biological blockchain operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{DropletNode, ElectroWettingGrid, Position2D};

/// Electro-wetting control parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectroWettingParams {
    pub max_voltage: f64,           // Maximum control voltage (V)
    pub force_coefficient: f64,      // Voltage to force conversion
    pub friction_coefficient: f64,   // Droplet movement friction
    pub surface_tension: f64,        // Surface tension effects
}

impl Default for ElectroWettingParams {
    fn default() -> Self {
        Self {
            max_voltage: 100.0,         // 100V max
            force_coefficient: 0.1,     // 0.1 N/V
            friction_coefficient: 0.05, // 5% friction
            surface_tension: 0.072,     // Water surface tension (N/m)
        }
    }
}

/// Create voltage pattern for droplet movement
pub fn create_movement_pattern(
    grid: &mut ElectroWettingGrid,
    target_droplet: &str,
    target_position: Position2D,
    params: &ElectroWettingParams
) -> Result<()> {
    let grid_size = grid.voltage_matrix.len();
    let target_x = (target_position.x / grid.pad_spacing_um) as usize;
    let target_y = (target_position.y / grid.pad_spacing_um) as usize;
    
    if target_x >= grid_size || target_y >= grid_size {
        return Err(anyhow::anyhow!("Target position outside grid bounds"));
    }

    // Reset voltage matrix
    for row in &mut grid.voltage_matrix {
        for cell in row {
            *cell = 0.0;
        }
    }

    // Create attractive voltage gradient toward target
    for x in 0..grid_size {
        for y in 0..grid_size {
            let distance = ((x as f64 - target_x as f64).powi(2) + 
                           (y as f64 - target_y as f64).powi(2)).sqrt();
            
            // Voltage decreases with distance from target
            let voltage = params.max_voltage * (-distance * 0.1).exp();
            grid.voltage_matrix[x][y] = voltage;
        }
    }

    debug!("⚡ Created movement pattern for {} to ({}, {})", 
           target_droplet, target_position.x, target_position.y);
    Ok(())
}

/// Calculate electro-wetting force on droplet
pub fn calculate_electro_wetting_force(
    droplet: &DropletNode,
    grid: &ElectroWettingGrid,
    params: &ElectroWettingParams
) -> (f64, f64) {
    let grid_x = (droplet.position.x / grid.pad_spacing_um).floor() as usize;
    let grid_y = (droplet.position.y / grid.pad_spacing_um).floor() as usize;
    
    if grid_x >= grid.voltage_matrix.len() || grid_y >= grid.voltage_matrix[0].len() {
        return (0.0, 0.0);
    }

    let voltage = grid.voltage_matrix[grid_x][grid_y];
    
    // Calculate voltage gradients for force direction
    let (grad_x, grad_y) = calculate_voltage_gradient(grid, grid_x, grid_y);
    
    // Force proportional to voltage and gradient
    let force_magnitude = voltage * params.force_coefficient;
    let force_x = force_magnitude * grad_x;
    let force_y = force_magnitude * grad_y;
    
    // Apply surface tension effects
    let tension_force = params.surface_tension * droplet.size_nanoliters / 1000.0;
    
    (force_x - tension_force * 0.1, force_y - tension_force * 0.1)
}

/// Calculate voltage gradient at grid position
fn calculate_voltage_gradient(
    grid: &ElectroWettingGrid,
    x: usize,
    y: usize
) -> (f64, f64) {
    let grid_size = grid.voltage_matrix.len();
    
    // Central difference for gradient calculation
    let grad_x = if x > 0 && x < grid_size - 1 {
        (grid.voltage_matrix[x + 1][y] - grid.voltage_matrix[x - 1][y]) / 2.0
    } else {
        0.0
    };
    
    let grad_y = if y > 0 && y < grid.voltage_matrix[0].len() - 1 {
        (grid.voltage_matrix[x][y + 1] - grid.voltage_matrix[x][y - 1]) / 2.0
    } else {
        0.0
    };
    
    (grad_x, grad_y)
}

/// Update droplet positions based on electro-wetting forces
pub async fn update_droplet_positions(
    droplets: &mut HashMap<String, DropletNode>,
    grid: &ElectroWettingGrid,
    params: &ElectroWettingParams,
    dt: f64
) -> Result<()> {
    for droplet in droplets.values_mut() {
        let (force_x, force_y) = calculate_electro_wetting_force(droplet, grid, params);
        
        // Update velocity (F = ma, assume unit mass)
        droplet.position.velocity_x += force_x * dt;
        droplet.position.velocity_y += force_y * dt;
        
        // Apply friction
        droplet.position.velocity_x *= 1.0 - params.friction_coefficient;
        droplet.position.velocity_y *= 1.0 - params.friction_coefficient;
        
        // Update position
        droplet.position.x += droplet.position.velocity_x * dt;
        droplet.position.y += droplet.position.velocity_y * dt;
        
        // Keep droplets within grid bounds
        droplet.position.x = droplet.position.x.max(0.0).min(grid.grid_size_mm);
        droplet.position.y = droplet.position.y.max(0.0).min(grid.grid_size_mm);
    }
    
    Ok(())
}

/// Create voltage pattern for droplet merging
pub fn create_merging_pattern(
    grid: &mut ElectroWettingGrid,
    droplet1_id: &str,
    droplet2_id: &str,
    merge_position: Position2D,
    params: &ElectroWettingParams
) -> Result<()> {
    // Create convergence pattern at merge position
    let grid_size = grid.voltage_matrix.len();
    let merge_x = (merge_position.x / grid.pad_spacing_um) as usize;
    let merge_y = (merge_position.y / grid.pad_spacing_um) as usize;
    
    // Reset voltage matrix
    for row in &mut grid.voltage_matrix {
        for cell in row {
            *cell = 0.0;
        }
    }
    
    // Create high voltage at merge point
    if merge_x < grid_size && merge_y < grid.voltage_matrix[0].len() {
        grid.voltage_matrix[merge_x][merge_y] = params.max_voltage;
        
        // Create gradient field pointing toward merge point
        for x in 0..grid_size {
            for y in 0..grid.voltage_matrix[0].len() {
                if x != merge_x || y != merge_y {
                    let distance = ((x as f64 - merge_x as f64).powi(2) + 
                                   (y as f64 - merge_y as f64).powi(2)).sqrt();
                    let voltage = params.max_voltage * 0.8 * (-distance * 0.2).exp();
                    grid.voltage_matrix[x][y] = voltage;
                }
            }
        }
    }
    
    info!("🔗 Created merging pattern for {} and {} at ({}, {})",
          droplet1_id, droplet2_id, merge_position.x, merge_position.y);
    Ok(())
}

/// Create voltage pattern for droplet division
pub fn create_division_pattern(
    grid: &mut ElectroWettingGrid,
    parent_droplet_id: &str,
    division_positions: (Position2D, Position2D),
    params: &ElectroWettingParams
) -> Result<()> {
    let grid_size = grid.voltage_matrix.len();
    
    // Reset voltage matrix
    for row in &mut grid.voltage_matrix {
        for cell in row {
            *cell = 0.0;
        }
    }
    
    // Create two high-voltage regions for daughter droplets
    let pos1_x = (division_positions.0.x / grid.pad_spacing_um) as usize;
    let pos1_y = (division_positions.0.y / grid.pad_spacing_um) as usize;
    let pos2_x = (division_positions.1.x / grid.pad_spacing_um) as usize;
    let pos2_y = (division_positions.1.y / grid.pad_spacing_um) as usize;
    
    if pos1_x < grid_size && pos1_y < grid.voltage_matrix[0].len() {
        grid.voltage_matrix[pos1_x][pos1_y] = params.max_voltage * 0.9;
    }
    
    if pos2_x < grid_size && pos2_y < grid.voltage_matrix[0].len() {
        grid.voltage_matrix[pos2_x][pos2_y] = params.max_voltage * 0.9;
    }
    
    // Create repulsive field in between
    let mid_x = (pos1_x + pos2_x) / 2;
    let mid_y = (pos1_y + pos2_y) / 2;
    
    if mid_x < grid_size && mid_y < grid.voltage_matrix[0].len() {
        grid.voltage_matrix[mid_x][mid_y] = -params.max_voltage * 0.3; // Repulsive
    }
    
    info!("✂️ Created division pattern for {} at ({}, {}) and ({}, {})",
          parent_droplet_id, 
          division_positions.0.x, division_positions.0.y,
          division_positions.1.x, division_positions.1.y);
    Ok(())
}

/// Check if two droplets are close enough to merge
pub fn check_droplet_collision(
    droplet1: &DropletNode,
    droplet2: &DropletNode,
    collision_threshold: f64
) -> bool {
    let distance = ((droplet1.position.x - droplet2.position.x).powi(2) +
                   (droplet1.position.y - droplet2.position.y).powi(2)).sqrt();
    
    distance < collision_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voltage_gradient_calculation() {
        let mut grid = ElectroWettingGrid {
            grid_size_mm: 10.0,
            pad_spacing_um: 100.0,
            voltage_matrix: vec![vec![0.0; 10]; 10],
            active_droplets: vec![],
        };
        
        // Create simple gradient
        grid.voltage_matrix[5][5] = 100.0;
        grid.voltage_matrix[4][5] = 80.0;
        grid.voltage_matrix[6][5] = 60.0;
        
        let (grad_x, grad_y) = calculate_voltage_gradient(&grid, 5, 5);
        assert!(grad_x.abs() > 0.0); // Should have gradient
    }

    #[test]
    fn test_collision_detection() {
        let droplet1 = DropletNode {
            droplet_id: "test1".to_string(),
            position: Position2D { x: 5.0, y: 5.0, velocity_x: 0.0, velocity_y: 0.0 },
            dna_data: crate::DNABlockchain {
                chain_length: 1,
                genesis_hash: "test".to_string(),
                latest_block_hash: "test".to_string(),
                total_mass_picograms: 1.0,
                synthesis_history: vec![],
            },
            energy_level: 1.0,
            size_nanoliters: 10.0,
        };
        
        let droplet2 = DropletNode {
            droplet_id: "test2".to_string(),
            position: Position2D { x: 5.1, y: 5.1, velocity_x: 0.0, velocity_y: 0.0 },
            dna_data: crate::DNABlockchain {
                chain_length: 1,
                genesis_hash: "test".to_string(),
                latest_block_hash: "test".to_string(),
                total_mass_picograms: 1.0,
                synthesis_history: vec![],
            },
            energy_level: 1.0,
            size_nanoliters: 10.0,
        };
        
        assert!(check_droplet_collision(&droplet1, &droplet2, 0.2)); // Close enough
        assert!(!check_droplet_collision(&droplet1, &droplet2, 0.05)); // Too far
    }
}