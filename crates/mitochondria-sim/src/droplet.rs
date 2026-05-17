/// Water-robot droplet management and physics
///
/// This module handles the core droplet functionality including movement,
/// energy management, and biological processes.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::{DropletNode, Position2D, DNABlockchain, DNASynthesisEvent, SimulationConfig};

/// Create a genesis droplet for the initial population
pub async fn create_genesis_droplet(id: usize, config: &SimulationConfig) -> Result<DropletNode> {
    let droplet_id = format!("droplet_{:04}", id);
    
    // Initialize with random position on the electro-wetting grid
    let position = Position2D {
        x: (id as f64 * 0.1) % config.grid_size_mm,
        y: (id as f64 * 0.15) % config.grid_size_mm,
        velocity_x: 0.0,
        velocity_y: 0.0,
    };

    // Initialize genesis DNA blockchain for this droplet
    let genesis_hash = format!("genesis_dna_{:08x}", id);
    let dna_data = DNABlockchain {
        chain_length: 1,
        genesis_hash: genesis_hash.clone(),
        latest_block_hash: genesis_hash,
        total_mass_picograms: 1.0, // Start with 1 picogram of DNA
        synthesis_history: vec![
            DNASynthesisEvent {
                block_height: 0,
                sequence_added: "ATGCGCGCATAGCTAG".to_string(), // Genesis sequence
                synthesis_time_ms: 0,
                energy_cost: 0.5,
                synthesized_at: Utc::now(),
            }
        ],
    };

    let droplet = DropletNode {
        droplet_id: droplet_id.clone(),
        position,
        dna_data,
        energy_level: 1.0, // Start fully charged
        size_nanoliters: 10.0, // 10 nL starting size
        tor_connection_id: format!("tor_circuit_{}", id % 4),
        last_consensus_vote: None,
        replication_readiness: 0.0,
    };

    debug!("💧 Created genesis droplet: {}", droplet_id);
    Ok(droplet)
}

/// Update droplet physics and biological processes
pub async fn update_droplet_physics(droplet: &mut DropletNode, dt: f64) -> Result<()> {
    // Update position based on velocity
    droplet.position.x += droplet.position.velocity_x * dt;
    droplet.position.y += droplet.position.velocity_y * dt;

    // Apply energy decay
    droplet.energy_level *= (1.0 - 0.01 * dt); // 1% decay per time unit
    
    // DNA synthesis if energy is sufficient
    if droplet.energy_level > 0.5 {
        synthesize_dna_block(droplet).await?;
    }

    // Check for droplet division
    if droplet.size_nanoliters > 100.0 && droplet.energy_level > 0.8 {
        info!("🔬 Droplet {} ready for binary fission", droplet.droplet_id);
    }

    Ok(())
}

/// Synthesize a new DNA block for blockchain operations
pub async fn synthesize_dna_block(droplet: &mut DropletNode) -> Result<()> {
    if droplet.energy_level < 0.3 {
        warn!("⚡ Insufficient energy for DNA synthesis in {}", droplet.droplet_id);
        return Ok(());
    }

    let new_block_height = droplet.dna_data.chain_length as u64;
    let synthesis_energy_cost = 0.2 * (1.0 + new_block_height as f64 * 0.01);
    
    // Generate DNA sequence encoding blockchain data
    let dna_sequence = generate_dna_sequence(new_block_height);
    
    let synthesis_event = DNASynthesisEvent {
        block_height: new_block_height,
        sequence_added: dna_sequence,
        synthesis_time_ms: 150 + (new_block_height * 10), // Slower as chain grows
        energy_cost: synthesis_energy_cost,
        synthesized_at: Utc::now(),
    };

    // Update droplet state
    droplet.energy_level -= synthesis_energy_cost;
    droplet.dna_data.chain_length += 1;
    droplet.dna_data.total_mass_picograms += synthesis_energy_cost * 2.0;
    droplet.dna_data.latest_block_hash = format!("block_{:08x}", new_block_height);
    droplet.dna_data.synthesis_history.push(synthesis_event);
    droplet.size_nanoliters += synthesis_energy_cost * 5.0; // Growth from DNA synthesis

    debug!("🧬 DNA block {} synthesized by {}", new_block_height, droplet.droplet_id);
    Ok(())
}

/// Generate DNA sequence for blockchain encoding
fn generate_dna_sequence(block_height: u64) -> String {
    // Simple encoding: block height -> DNA bases
    let bases = ["A", "T", "G", "C"];
    let mut sequence = String::new();
    
    let mut height = block_height;
    for _ in 0..16 {
        sequence.push_str(bases[(height % 4) as usize]);
        height /= 4;
    }
    
    sequence
}

/// Calculate electro-wetting force on droplet
pub fn calculate_electro_wetting_force(
    droplet: &DropletNode, 
    voltage_matrix: &[Vec<f64>],
    grid_spacing: f64
) -> (f64, f64) {
    let grid_x = (droplet.position.x / grid_spacing) as usize;
    let grid_y = (droplet.position.y / grid_spacing) as usize;
    
    if grid_x >= voltage_matrix.len() || grid_y >= voltage_matrix[0].len() {
        return (0.0, 0.0);
    }

    let voltage = voltage_matrix[grid_x][grid_y];
    let force_magnitude = voltage * 0.1; // Simplified electro-wetting force
    
    // Force direction based on voltage gradient
    let force_x = force_magnitude * (droplet.position.x.sin());
    let force_y = force_magnitude * (droplet.position.y.cos());
    
    (force_x, force_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_genesis_droplet_creation() {
        let config = SimulationConfig::default();
        let droplet = create_genesis_droplet(0, &config).await.unwrap();
        
        assert_eq!(droplet.droplet_id, "droplet_0000");
        assert_eq!(droplet.dna_data.chain_length, 1);
        assert!(droplet.energy_level > 0.9);
    }

    #[tokio::test]
    async fn test_dna_synthesis() {
        let config = SimulationConfig::default();
        let mut droplet = create_genesis_droplet(1, &config).await.unwrap();
        
        let initial_chain_length = droplet.dna_data.chain_length;
        synthesize_dna_block(&mut droplet).await.unwrap();
        
        assert_eq!(droplet.dna_data.chain_length, initial_chain_length + 1);
        assert!(droplet.energy_level < 1.0); // Energy consumed
    }

    #[test]
    fn test_dna_sequence_generation() {
        let sequence = generate_dna_sequence(42);
        assert_eq!(sequence.len(), 16);
        assert!(sequence.chars().all(|c| matches!(c, 'A' | 'T' | 'G' | 'C')));
    }
}