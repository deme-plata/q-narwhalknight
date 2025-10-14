//! # Q-Higgs-Simulator
//!
//! 3D Higgs field simulation with attosecond laser interaction.
//!
//! This crate provides tools for simulating the Higgs field dynamics,
//! including the Mexican hat potential, laser-field interactions, and
//! topological defect evolution.
//!
//! ## Core Features
//! - 3D scalar field simulation with finite-difference evolution
//! - Mexican hat potential with controllable parameters
//! - Attosecond laser pulse interaction
//! - Parallel processing optimized for multi-core CPUs
//! - VTK visualization export
//! - Real-time field monitoring

pub mod field;
pub mod evolution;
pub mod laser_interaction;
pub mod visualization;
pub mod constants;
pub mod factory;

pub use field::*;
pub use evolution::*;
pub use laser_interaction::*;
pub use visualization::*;
pub use constants::*;
pub use factory::*;

use anyhow::Result;

/// Main simulator struct coordinating all components
pub struct HiggsSimulator {
    pub field: ScalarField3D,
    pub evolution: FieldEvolution,
    pub laser: Option<LaserPulse>,
    pub constants: PhysicalConstants,
}

impl HiggsSimulator {
    /// Create a new Higgs field simulator
    pub fn new(resolution: usize, size_nm: f64) -> Self {
        let constants = PhysicalConstants::standard_model();
        let field = ScalarField3D::new(resolution, size_nm);
        let evolution = FieldEvolution::new(constants.clone());

        Self {
            field,
            evolution,
            laser: None,
            constants,
        }
    }

    /// Initialize field with vacuum expectation value
    pub fn initialize_vacuum(&mut self) {
        self.field.fill_uniform(self.constants.vacuum_expectation_value());
    }

    /// Add small perturbation for testing stability
    pub fn add_perturbation(&mut self, amplitude: f64, wavelength_nm: f64) {
        self.field.add_wave_perturbation(amplitude, wavelength_nm);
    }

    /// Set laser pulse for interaction
    pub fn set_laser(&mut self, laser: LaserPulse) {
        self.laser = Some(laser);
    }

    /// Evolve field for one time step
    pub fn step(&mut self, dt_as: f64) -> Result<()> {
        // Apply laser interaction if present
        if let Some(ref laser) = self.laser {
            self.evolution.apply_laser_interaction(&mut self.field, laser, dt_as)?;
        }

        // Evolve field according to Klein-Gordon equation with Mexican hat potential
        self.evolution.evolve_step(&mut self.field, dt_as)?;

        Ok(())
    }

    /// Run simulation for specified duration
    pub async fn simulate(&mut self, duration_as: f64, dt_as: f64) -> Result<SimulationResult> {
        let steps = (duration_as / dt_as) as usize;
        let mut metrics = SimulationMetrics::new();

        for step in 0..steps {
            self.step(dt_as)?;

            // Collect metrics every 100 steps
            if step % 100 == 0 {
                metrics.record_step(step, &self.field);
            }
        }

        Ok(SimulationResult {
            final_field: self.field.clone(),
            metrics,
        })
    }
}

/// Results from a simulation run
#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub final_field: ScalarField3D,
    pub metrics: SimulationMetrics,
}

/// Metrics collected during simulation
#[derive(Clone, Debug)]
pub struct SimulationMetrics {
    pub total_energy: Vec<f64>,
    pub max_field_value: Vec<f64>,
    pub min_field_value: Vec<f64>,
    pub steps: Vec<usize>,
}

impl SimulationMetrics {
    pub fn new() -> Self {
        Self {
            total_energy: Vec::new(),
            max_field_value: Vec::new(),
            min_field_value: Vec::new(),
            steps: Vec::new(),
        }
    }

    pub fn record_step(&mut self, step: usize, field: &ScalarField3D) {
        self.steps.push(step);
        self.total_energy.push(field.total_energy());
        self.max_field_value.push(field.max_value());
        self.min_field_value.push(field.min_value());
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
