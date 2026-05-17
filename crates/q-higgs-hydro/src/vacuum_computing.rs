//! Vacuum Computing: Direct manipulation of spacetime for computation
//!
//! Inspired by Seth Lloyd's vision of the universe as a quantum computer,
//! this module implements computation at the level of the vacuum itself,
//! treating empty space as a computational substrate.

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    f64::consts::{E, PI},
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

use crate::{PhysicalConstants, HiggsBit, QuantumDroplet};

/// The Vacuum State Computer - computation using spacetime itself
#[derive(Debug)]
pub struct VacuumStateComputer {
    /// 3D grid of vacuum states
    vacuum_grid: Vec<Vec<Vec<VacuumCell>>>,
    /// Grid dimensions
    grid_size: (usize, usize, usize),
    /// Spatial resolution in meters  
    spatial_resolution: f64,
    /// Temporal resolution in attoseconds
    temporal_resolution: f64,
    /// Physical constants
    constants: PhysicalConstants,
    /// Active computation threads in vacuum
    active_computations: HashMap<String, VacuumComputation>,
}

/// A single cell of vacuum spacetime used for computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumCell {
    /// Local spacetime curvature (Einstein tensor component)
    pub curvature: Matrix3<f64>,
    /// Vacuum energy density (cosmological constant contribution)
    pub energy_density: f64,
    /// Quantum field fluctuations (virtual particle density)
    pub quantum_fluctuations: Complex64,
    /// Information density (Lloyd bound)
    pub information_density: f64,
    /// Entanglement connections to neighboring cells
    pub entanglement_links: Vec<(usize, usize, usize, f64)>, // (x,y,z,strength)
    /// Last computation timestamp
    pub last_computed: Option<Instant>,
    /// Computational state (qubit representation)
    pub quantum_state: Complex64,
}

impl Default for VacuumCell {
    fn default() -> Self {
        Self {
            curvature: Matrix3::zeros(),
            energy_density: -1.0e-29, // Vacuum energy (J/m³)
            quantum_fluctuations: Complex64::new(0.0, 0.0),
            information_density: 0.0,
            entanglement_links: Vec::new(),
            last_computed: None,
            quantum_state: Complex64::new(1.0, 0.0), // |0⟩ state
        }
    }
}

/// A computation running in the vacuum
#[derive(Debug, Clone)]
pub struct VacuumComputation {
    /// Unique computation identifier
    pub id: String,
    /// Spatial region occupied by computation
    pub computation_region: ((usize, usize, usize), (usize, usize, usize)),
    /// Current computation step
    pub current_step: usize,
    /// Total steps in computation
    pub total_steps: usize,
    /// Start time
    pub start_time: Instant,
    /// Expected completion time
    pub expected_duration: Duration,
    /// Computation type
    pub computation_type: VacuumComputationType,
    /// Intermediate results
    pub results: Vec<VacuumComputationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VacuumComputationType {
    /// Quantum simulation using vacuum fluctuations
    QuantumSimulation { system_size: usize },
    /// Cryptographic operation using spacetime entanglement
    CryptographicOperation { key_size: usize },
    /// Machine learning using vacuum state patterns
    MachineLearning { training_samples: usize },
    /// Optimization using spacetime geometry
    OptimizationProblem { variables: usize },
    /// Lloyd universe computation
    UniverseSimulation { particle_count: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumComputationResult {
    pub step: usize,
    pub result_data: Vec<f64>,
    pub computation_energy: f64,
    pub information_processed: f64,
    pub vacuum_entropy: f64,
}

impl VacuumStateComputer {
    /// Create new vacuum state computer with given grid dimensions
    pub fn new(
        grid_size: (usize, usize, usize),
        spatial_resolution: f64,
        temporal_resolution: f64,
    ) -> Result<Self> {
        info!(
            "Initializing vacuum state computer: {:?} grid, {:.2e}m resolution",
            grid_size, spatial_resolution
        );

        let constants = PhysicalConstants::default();
        let mut vacuum_grid = Vec::with_capacity(grid_size.0);

        // Initialize 3D vacuum grid
        for x in 0..grid_size.0 {
            let mut y_layer = Vec::with_capacity(grid_size.1);
            for y in 0..grid_size.1 {
                let mut z_layer = Vec::with_capacity(grid_size.2);
                for z in 0..grid_size.2 {
                    let mut cell = VacuumCell::default();
                    
                    // Initialize with quantum vacuum fluctuations
                    cell.quantum_fluctuations = Complex64::new(
                        (x + y + z) as f64 * 1e-12, // Deterministic but complex pattern
                        (x * y + z) as f64 * 1e-12,
                    );
                    
                    // Set initial information density using Lloyd bound
                    cell.information_density = (spatial_resolution / constants.planck_constant).ln();
                    
                    z_layer.push(cell);
                }
                y_layer.push(z_layer);
            }
            vacuum_grid.push(y_layer);
        }

        info!("Vacuum grid initialized with {} total cells", 
               grid_size.0 * grid_size.1 * grid_size.2);

        Ok(Self {
            vacuum_grid,
            grid_size,
            spatial_resolution,
            temporal_resolution,
            constants,
            active_computations: HashMap::new(),
        })
    }

    /// Initialize entanglement network between vacuum cells
    pub async fn initialize_entanglement_network(&mut self) -> Result<()> {
        info!("Initializing vacuum entanglement network");

        let mut total_links = 0;

        for x in 0..self.grid_size.0 {
            for y in 0..self.grid_size.1 {
                for z in 0..self.grid_size.2 {
                    // Create entanglement with neighboring cells
                    let neighbors = self.get_neighboring_cells(x, y, z);
                    
                    for (nx, ny, nz) in neighbors {
                        if self.should_create_entanglement(x, y, z, nx, ny, nz) {
                            let strength = self.calculate_entanglement_strength(x, y, z, nx, ny, nz);
                            self.vacuum_grid[x][y][z].entanglement_links.push((nx, ny, nz, strength));
                            total_links += 1;
                        }
                    }
                }
            }
        }

        info!("Created {} entanglement links in vacuum network", total_links);
        Ok(())
    }

    /// Get neighboring cells for entanglement
    fn get_neighboring_cells(&self, x: usize, y: usize, z: usize) -> Vec<(usize, usize, usize)> {
        let mut neighbors = Vec::new();
        
        for dx in -1..=1i32 {
            for dy in -1..=1i32 {
                for dz in -1..=1i32 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue; // Skip self
                    }
                    
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    
                    if nx >= 0 && ny >= 0 && nz >= 0 
                        && (nx as usize) < self.grid_size.0 
                        && (ny as usize) < self.grid_size.1 
                        && (nz as usize) < self.grid_size.2 {
                        neighbors.push((nx as usize, ny as usize, nz as usize));
                    }
                }
            }
        }
        
        neighbors
    }

    /// Determine if entanglement should be created between cells
    fn should_create_entanglement(&self, x1: usize, y1: usize, z1: usize, x2: usize, y2: usize, z2: usize) -> bool {
        // Create entanglement based on distance and quantum coherence
        let distance = ((x2 as f64 - x1 as f64).powi(2) + 
                       (y2 as f64 - y1 as f64).powi(2) + 
                       (z2 as f64 - z1 as f64).powi(2)).sqrt();
        
        let coherence_length = 10.0; // In grid units
        distance <= coherence_length
    }

    /// Calculate entanglement strength between cells
    fn calculate_entanglement_strength(&self, x1: usize, y1: usize, z1: usize, x2: usize, y2: usize, z2: usize) -> f64 {
        let distance = ((x2 as f64 - x1 as f64).powi(2) + 
                       (y2 as f64 - y1 as f64).powi(2) + 
                       (z2 as f64 - z1 as f64).powi(2)).sqrt();
        
        // Exponential decay with distance
        (-distance / 5.0).exp()
    }

    /// Start a new computation in the vacuum
    pub async fn start_vacuum_computation(
        &mut self,
        computation_type: VacuumComputationType,
        region: ((usize, usize, usize), (usize, usize, usize)),
    ) -> Result<String> {
        let computation_id = format!("vacuum_comp_{}", self.active_computations.len());
        
        let (total_steps, expected_duration) = match &computation_type {
            VacuumComputationType::QuantumSimulation { system_size } => {
                (*system_size * 100, Duration::from_millis(*system_size as u64))
            }
            VacuumComputationType::CryptographicOperation { key_size } => {
                (*key_size * 10, Duration::from_micros(*key_size as u64))
            }
            VacuumComputationType::MachineLearning { training_samples } => {
                (*training_samples, Duration::from_millis(*training_samples as u64 / 10))
            }
            VacuumComputationType::OptimizationProblem { variables } => {
                (*variables * *variables, Duration::from_millis(*variables as u64 * 10))
            }
            VacuumComputationType::UniverseSimulation { particle_count } => {
                (*particle_count * 1000, Duration::from_secs(*particle_count as u64))
            }
        };

        let computation = VacuumComputation {
            id: computation_id.clone(),
            computation_region: region,
            current_step: 0,
            total_steps,
            start_time: Instant::now(),
            expected_duration,
            computation_type,
            results: Vec::new(),
        };

        info!(
            "Starting vacuum computation '{}' of type {:?} in region {:?}",
            computation_id, computation.computation_type, region
        );

        self.active_computations.insert(computation_id.clone(), computation);
        Ok(computation_id)
    }

    /// Execute one step of a vacuum computation
    pub async fn step_vacuum_computation(&mut self, computation_id: &str) -> Result<bool> {
        let computation = self.active_computations
            .get_mut(computation_id)
            .context("Computation not found")?
            .clone();

        if computation.current_step >= computation.total_steps {
            return Ok(true); // Computation complete
        }

        let result = match &computation.computation_type {
            VacuumComputationType::QuantumSimulation { system_size } => {
                self.step_quantum_simulation(&computation, *system_size).await?
            }
            VacuumComputationType::CryptographicOperation { key_size } => {
                self.step_cryptographic_operation(&computation, *key_size).await?
            }
            VacuumComputationType::MachineLearning { training_samples } => {
                self.step_machine_learning(&computation, *training_samples).await?
            }
            VacuumComputationType::OptimizationProblem { variables } => {
                self.step_optimization(&computation, *variables).await?
            }
            VacuumComputationType::UniverseSimulation { particle_count } => {
                self.step_universe_simulation(&computation, *particle_count).await?
            }
        };

        // Update computation state
        if let Some(comp) = self.active_computations.get_mut(computation_id) {
            comp.current_step += 1;
            comp.results.push(result);
        }

        let is_complete = computation.current_step + 1 >= computation.total_steps;
        
        if is_complete {
            info!("Vacuum computation '{}' completed", computation_id);
        }

        Ok(is_complete)
    }

    /// Execute quantum simulation step in vacuum
    async fn step_quantum_simulation(
        &mut self,
        computation: &VacuumComputation,
        system_size: usize,
    ) -> Result<VacuumComputationResult> {
        let ((x1, y1, z1), (x2, y2, z2)) = computation.computation_region;
        let mut computation_energy = 0.0;
        let mut information_processed = 0.0;
        let mut result_data = Vec::new();

        // Simulate quantum evolution in vacuum cells
        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &mut self.vacuum_grid[x][y][z];
                    
                    // Evolve quantum state using Schrodinger-like evolution
                    let evolution_phase = computation.current_step as f64 * 0.1;
                    let evolution_operator = Complex64::new(
                        evolution_phase.cos(),
                        evolution_phase.sin(),
                    );
                    
                    cell.quantum_state *= evolution_operator;
                    cell.last_computed = Some(Instant::now());
                    
                    // Calculate energy and information
                    let cell_energy = cell.quantum_state.norm_sqr() * cell.energy_density.abs();
                    computation_energy += cell_energy;
                    information_processed += cell.information_density;
                    
                    // Record measurement result
                    result_data.push(cell.quantum_state.norm_sqr());
                }
            }
        }

        // Calculate vacuum entropy change
        let vacuum_entropy = self.calculate_vacuum_entropy(computation.computation_region).await?;

        Ok(VacuumComputationResult {
            step: computation.current_step,
            result_data,
            computation_energy,
            information_processed,
            vacuum_entropy,
        })
    }

    /// Execute cryptographic operation step in vacuum
    async fn step_cryptographic_operation(
        &mut self,
        computation: &VacuumComputation,
        key_size: usize,
    ) -> Result<VacuumComputationResult> {
        let ((x1, y1, z1), (x2, y2, z2)) = computation.computation_region;
        let mut result_data = Vec::new();
        let mut computation_energy = 0.0;

        // Use vacuum entanglement for quantum key generation
        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &mut self.vacuum_grid[x][y][z];
                    
                    // Extract random bits from quantum fluctuations
                    let random_bit = if cell.quantum_fluctuations.re > 0.0 { 1.0 } else { 0.0 };
                    result_data.push(random_bit);
                    
                    // Update fluctuations to prevent reuse
                    cell.quantum_fluctuations *= Complex64::new(0.9, 0.1);
                    computation_energy += cell.energy_density.abs() * 1e-12;
                    
                    if result_data.len() >= key_size {
                        break;
                    }
                }
                if result_data.len() >= key_size {
                    break;
                }
            }
            if result_data.len() >= key_size {
                break;
            }
        }

        let vacuum_entropy = self.calculate_vacuum_entropy(computation.computation_region).await?;

        Ok(VacuumComputationResult {
            step: computation.current_step,
            result_data,
            computation_energy,
            information_processed: result_data.len() as f64,
            vacuum_entropy,
        })
    }

    /// Execute machine learning step in vacuum  
    async fn step_machine_learning(
        &mut self,
        computation: &VacuumComputation,
        training_samples: usize,
    ) -> Result<VacuumComputationResult> {
        let ((x1, y1, z1), (x2, y2, z2)) = computation.computation_region;
        let mut result_data = Vec::new();
        let mut computation_energy = 0.0;
        let mut information_processed = 0.0;

        // Use vacuum patterns for feature learning
        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &mut self.vacuum_grid[x][y][z];
                    
                    // Extract features from local vacuum structure
                    let feature = cell.information_density * cell.quantum_state.norm();
                    result_data.push(feature);
                    
                    // Update information density based on learning
                    let learning_rate = 0.01;
                    cell.information_density *= 1.0 + learning_rate * feature.signum();
                    
                    computation_energy += feature.abs();
                    information_processed += cell.information_density;
                }
            }
        }

        let vacuum_entropy = self.calculate_vacuum_entropy(computation.computation_region).await?;

        Ok(VacuumComputationResult {
            step: computation.current_step,
            result_data,
            computation_energy,
            information_processed,
            vacuum_entropy,
        })
    }

    /// Execute optimization step in vacuum
    async fn step_optimization(
        &mut self,
        computation: &VacuumComputation,
        variables: usize,
    ) -> Result<VacuumComputationResult> {
        let ((x1, y1, z1), (x2, y2, z2)) = computation.computation_region;
        let mut result_data = Vec::new();
        let mut computation_energy = 0.0;

        // Use vacuum geometry for optimization landscape
        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &mut self.vacuum_grid[x][y][z];
                    
                    // Calculate local cost function from curvature
                    let cost = cell.curvature.trace() + cell.energy_density;
                    result_data.push(cost);
                    
                    // Gradient descent in vacuum space
                    let gradient = cost * 0.001; // Small step
                    cell.energy_density -= gradient;
                    
                    computation_energy += cost.abs();
                    
                    if result_data.len() >= variables {
                        break;
                    }
                }
                if result_data.len() >= variables {
                    break;
                }
            }
            if result_data.len() >= variables {
                break;
            }
        }

        let vacuum_entropy = self.calculate_vacuum_entropy(computation.computation_region).await?;

        Ok(VacuumComputationResult {
            step: computation.current_step,
            result_data,
            computation_energy,
            information_processed: variables as f64,
            vacuum_entropy,
        })
    }

    /// Execute universe simulation step (Seth Lloyd's universe computer)
    async fn step_universe_simulation(
        &mut self,
        computation: &VacuumComputation,
        particle_count: usize,
    ) -> Result<VacuumComputationResult> {
        let ((x1, y1, z1), (x2, y2, z2)) = computation.computation_region;
        let mut result_data = Vec::new();
        let mut computation_energy = 0.0;
        let mut information_processed = 0.0;

        // Simulate fundamental interactions in vacuum
        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &mut self.vacuum_grid[x][y][z];
                    
                    // Simulate particle interactions through vacuum fluctuations
                    let interaction_strength = cell.quantum_fluctuations.norm_sqr();
                    let particle_density = interaction_strength * particle_count as f64;
                    
                    result_data.push(particle_density);
                    
                    // Update vacuum based on particle interactions
                    cell.quantum_fluctuations *= Complex64::new(0.99, 0.01);
                    cell.energy_density += interaction_strength * 1e-15;
                    
                    computation_energy += interaction_strength;
                    information_processed += cell.information_density * interaction_strength;
                }
            }
        }

        let vacuum_entropy = self.calculate_vacuum_entropy(computation.computation_region).await?;

        Ok(VacuumComputationResult {
            step: computation.current_step,
            result_data,
            computation_energy,
            information_processed,
            vacuum_entropy,
        })
    }

    /// Calculate vacuum entropy in a region
    async fn calculate_vacuum_entropy(&self, region: ((usize, usize, usize), (usize, usize, usize))) -> Result<f64> {
        let ((x1, y1, z1), (x2, y2, z2)) = region;
        let mut entropy = 0.0;
        let mut total_cells = 0;

        for x in x1..=x2.min(self.grid_size.0 - 1) {
            for y in y1..=y2.min(self.grid_size.1 - 1) {
                for z in z1..=z2.min(self.grid_size.2 - 1) {
                    let cell = &self.vacuum_grid[x][y][z];
                    
                    // Calculate local entropy from quantum state
                    let p = cell.quantum_state.norm_sqr();
                    if p > 0.0 && p < 1.0 {
                        entropy -= p * p.ln() + (1.0 - p) * (1.0 - p).ln();
                    }
                    
                    total_cells += 1;
                }
            }
        }

        if total_cells > 0 {
            entropy /= total_cells as f64;
        }

        Ok(entropy)
    }

    /// Get computation status
    pub fn get_computation_status(&self, computation_id: &str) -> Option<(usize, usize, f64)> {
        self.active_computations.get(computation_id).map(|comp| {
            let progress = comp.current_step as f64 / comp.total_steps as f64;
            (comp.current_step, comp.total_steps, progress)
        })
    }

    /// Get vacuum grid statistics
    pub async fn get_vacuum_statistics(&self) -> VacuumStatistics {
        let mut total_energy = 0.0;
        let mut total_information = 0.0;
        let mut total_entanglements = 0;
        let mut active_cells = 0;

        for x in 0..self.grid_size.0 {
            for y in 0..self.grid_size.1 {
                for z in 0..self.grid_size.2 {
                    let cell = &self.vacuum_grid[x][y][z];
                    
                    total_energy += cell.energy_density.abs();
                    total_information += cell.information_density;
                    total_entanglements += cell.entanglement_links.len();
                    
                    if cell.last_computed.is_some() {
                        active_cells += 1;
                    }
                }
            }
        }

        let total_cells = self.grid_size.0 * self.grid_size.1 * self.grid_size.2;

        VacuumStatistics {
            total_cells,
            active_cells,
            total_energy,
            average_energy: total_energy / total_cells as f64,
            total_information,
            average_information: total_information / total_cells as f64,
            total_entanglements,
            active_computations: self.active_computations.len(),
            grid_size: self.grid_size,
            spatial_resolution: self.spatial_resolution,
            temporal_resolution: self.temporal_resolution,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumStatistics {
    pub total_cells: usize,
    pub active_cells: usize,
    pub total_energy: f64,
    pub average_energy: f64,
    pub total_information: f64,
    pub average_information: f64,
    pub total_entanglements: usize,
    pub active_computations: usize,
    pub grid_size: (usize, usize, usize),
    pub spatial_resolution: f64,
    pub temporal_resolution: f64,
}

/// Integration with quantum droplets for hybrid vacuum-matter computation
impl VacuumStateComputer {
    /// Place a quantum droplet in the vacuum grid
    pub async fn place_droplet_in_vacuum(
        &mut self,
        droplet: &QuantumDroplet,
        position: (usize, usize, usize),
    ) -> Result<()> {
        let (x, y, z) = position;
        
        if x >= self.grid_size.0 || y >= self.grid_size.1 || z >= self.grid_size.2 {
            return Err(anyhow::anyhow!("Position out of vacuum grid bounds"));
        }

        let cell = &mut self.vacuum_grid[x][y][z];
        
        // Droplet influences local vacuum properties
        cell.energy_density += droplet.lloyd_state.computation_energy * 1e-12;
        cell.information_density += droplet.total_lloyd_entropy();
        
        // Create quantum correlation between droplet and vacuum
        let correlation = Complex64::new(
            droplet.current_laser_phase.cos(),
            droplet.current_laser_phase.sin(),
        );
        cell.quantum_state *= correlation;

        info!("Placed droplet {} in vacuum at ({}, {}, {})", 
              hex::encode(droplet.id), x, y, z);

        Ok(())
    }

    /// Simulate interaction between droplets through vacuum medium
    pub async fn simulate_droplet_vacuum_interaction(
        &mut self,
        droplet1_pos: (usize, usize, usize),
        droplet2_pos: (usize, usize, usize),
    ) -> Result<f64> {
        let (x1, y1, z1) = droplet1_pos;
        let (x2, y2, z2) = droplet2_pos;

        // Calculate interaction through vacuum path
        let distance = ((x2 as f64 - x1 as f64).powi(2) + 
                       (y2 as f64 - y1 as f64).powi(2) + 
                       (z2 as f64 - z1 as f64).powi(2)).sqrt();

        let mut interaction_strength = 0.0;

        // Sample vacuum cells along the path
        let steps = distance.ceil() as usize;
        for step in 0..steps {
            let t = step as f64 / steps as f64;
            let x = (x1 as f64 * (1.0 - t) + x2 as f64 * t) as usize;
            let y = (y1 as f64 * (1.0 - t) + y2 as f64 * t) as usize;
            let z = (z1 as f64 * (1.0 - t) + z2 as f64 * t) as usize;

            if x < self.grid_size.0 && y < self.grid_size.1 && z < self.grid_size.2 {
                let cell = &self.vacuum_grid[x][y][z];
                interaction_strength += cell.quantum_state.norm_sqr() * cell.energy_density.abs();
            }
        }

        interaction_strength /= steps as f64;

        debug!("Vacuum-mediated interaction strength: {:.2e}", interaction_strength);
        Ok(interaction_strength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vacuum_computer_creation() {
        let computer = VacuumStateComputer::new((10, 10, 10), 1e-15, 0.1).unwrap();
        assert_eq!(computer.grid_size, (10, 10, 10));
        assert_eq!(computer.spatial_resolution, 1e-15);
    }

    #[tokio::test]
    async fn test_entanglement_initialization() {
        let mut computer = VacuumStateComputer::new((5, 5, 5), 1e-15, 0.1).unwrap();
        computer.initialize_entanglement_network().await.unwrap();
        
        // Check that some entanglement links were created
        let mut total_links = 0;
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    total_links += computer.vacuum_grid[x][y][z].entanglement_links.len();
                }
            }
        }
        assert!(total_links > 0);
    }

    #[tokio::test]
    async fn test_vacuum_computation_start() {
        let mut computer = VacuumStateComputer::new((5, 5, 5), 1e-15, 0.1).unwrap();
        
        let computation_type = VacuumComputationType::QuantumSimulation { system_size: 10 };
        let region = ((0, 0, 0), (2, 2, 2));
        
        let computation_id = computer.start_vacuum_computation(computation_type, region).await.unwrap();
        assert!(computer.active_computations.contains_key(&computation_id));
    }

    #[tokio::test]
    async fn test_vacuum_statistics() {
        let computer = VacuumStateComputer::new((3, 3, 3), 1e-15, 0.1).unwrap();
        let stats = computer.get_vacuum_statistics().await;
        
        assert_eq!(stats.total_cells, 27);
        assert_eq!(stats.grid_size, (3, 3, 3));
        assert!(stats.average_energy < 0.0); // Vacuum energy is negative
    }
}