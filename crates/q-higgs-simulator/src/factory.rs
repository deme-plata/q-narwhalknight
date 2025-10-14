//! Higgs Field Simulation Factory - Industrial Scale Orchestration
//!
//! This module provides a factory pattern for managing multiple Higgs field simulations
//! with different parameters, running in parallel, with automated analysis and comparison.

use crate::{
    HiggsSimulator, LaserPulse, SimulationResult, VtkExporter, AsciiExporter,
    constants::PhysicalConstants,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::sync::Semaphore;
use std::sync::Arc;
use tracing::{info, debug, warn};

/// Simulation configuration template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Unique identifier for this simulation
    pub id: String,

    /// Grid resolution (e.g., 32, 64, 128)
    pub resolution: usize,

    /// Physical box size in nanometers
    pub size_nm: f64,

    /// Time step in attoseconds
    pub dt_as: f64,

    /// Total simulation duration in attoseconds
    pub duration_as: f64,

    /// Gaussian perturbation amplitude (GeV)
    pub perturbation_amplitude: f64,

    /// Gaussian perturbation width (nm)
    pub perturbation_width: f64,

    /// Laser configuration
    pub laser: LaserConfig,

    /// Output directory
    pub output_dir: PathBuf,
}

/// Laser pulse configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserConfig {
    /// Pulse duration in attoseconds
    pub duration_as: f64,

    /// Photon energy in eV
    pub photon_energy_ev: f64,

    /// Peak intensity in W/cm²
    pub peak_intensity: f64,
}

/// Factory for creating and managing multiple simulations
pub struct SimulationFactory {
    /// Maximum number of concurrent simulations
    max_concurrent: usize,

    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,

    /// Physical constants (shared across simulations)
    constants: PhysicalConstants,
}

impl SimulationFactory {
    /// Create a new simulation factory
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            constants: PhysicalConstants::standard_model(),
        }
    }

    /// Run a single simulation with the given configuration
    pub async fn run_simulation(&self, config: SimulationConfig) -> Result<SimulationOutput> {
        let _permit = self.semaphore.acquire().await
            .context("Failed to acquire semaphore permit")?;

        info!("Starting simulation: {} ({}³ grid, {} as)",
              config.id, config.resolution, config.duration_as);

        let start_time = std::time::Instant::now();

        // Create simulator
        let mut simulator = HiggsSimulator::new(config.resolution, config.size_nm);

        // Initialize field
        simulator.initialize_vacuum();
        simulator.field.add_gaussian_perturbation(
            config.perturbation_amplitude,
            config.perturbation_width,
        );

        // Configure laser
        let laser = LaserPulse::xuv_attosecond(
            config.laser.duration_as,
            config.laser.photon_energy_ev,
            config.laser.peak_intensity,
        );
        simulator.set_laser(laser);

        // Run simulation
        let result = simulator.simulate(config.duration_as, config.dt_as).await
            .context("Simulation failed")?;

        let elapsed = start_time.elapsed();

        // Create output directory
        std::fs::create_dir_all(&config.output_dir)
            .context("Failed to create output directory")?;

        // Export results
        self.export_results(&config, &result).await?;

        // Calculate performance metrics
        let total_grid_points = config.resolution.pow(3);
        let total_timesteps = (config.duration_as / config.dt_as) as usize;
        let total_updates = total_grid_points * total_timesteps;
        let updates_per_second = total_updates as f64 / elapsed.as_secs_f64();

        let output = SimulationOutput {
            config: config.clone(),
            result,
            elapsed_seconds: elapsed.as_secs_f64(),
            updates_per_second,
        };

        info!("Completed simulation: {} in {:.2}s ({:.2e} updates/sec)",
              config.id, elapsed.as_secs_f64(), updates_per_second);

        Ok(output)
    }

    /// Run multiple simulations in parallel
    pub async fn run_batch(&self, configs: Vec<SimulationConfig>) -> Result<Vec<SimulationOutput>> {
        info!("Starting batch of {} simulations (max {} concurrent)",
              configs.len(), self.max_concurrent);

        let mut handles = Vec::new();

        for config in configs {
            let factory = self.clone_for_task();
            let handle = tokio::spawn(async move {
                factory.run_simulation(config).await
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(output)) => results.push(output),
                Ok(Err(e)) => warn!("Simulation failed: {}", e),
                Err(e) => warn!("Task join error: {}", e),
            }
        }

        info!("Batch complete: {}/{} simulations succeeded",
              results.len(), results.len());

        Ok(results)
    }

    /// Export simulation results to files
    async fn export_results(&self, config: &SimulationConfig, result: &SimulationResult) -> Result<()> {
        let output_dir = &config.output_dir;

        // Export 3D field
        VtkExporter::export_scalar_field(
            &result.final_field,
            output_dir.join("higgs_field_final.vtk"),
            "phi",
        ).context("Failed to export 3D field")?;

        // Export central slice
        let center = config.resolution / 2;
        VtkExporter::export_2d_slice(
            &result.final_field,
            center,
            output_dir.join("higgs_slice_xy.vtk"),
            "phi",
        ).context("Failed to export slice")?;

        // Export line profile
        AsciiExporter::export_line_profile(
            &result.final_field,
            output_dir.join("line_profile.txt"),
        ).context("Failed to export line profile")?;

        // Export metrics
        self.export_metrics(&result, output_dir.join("metrics.txt"))?;

        debug!("Exported all results for simulation: {}", config.id);

        Ok(())
    }

    /// Export metrics to text file
    fn export_metrics(&self, result: &SimulationResult, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "# Higgs Field Simulation Metrics")?;
        writeln!(file, "# step total_energy max_field min_field")?;

        for i in 0..result.metrics.steps.len() {
            writeln!(
                file,
                "{} {:.6e} {:.6e} {:.6e}",
                result.metrics.steps[i],
                result.metrics.total_energy[i],
                result.metrics.max_field_value[i],
                result.metrics.min_field_value[i]
            )?;
        }

        Ok(())
    }

    /// Clone factory for use in async task
    fn clone_for_task(&self) -> Self {
        Self {
            max_concurrent: self.max_concurrent,
            semaphore: self.semaphore.clone(),
            constants: self.constants.clone(),
        }
    }

    /// Generate a parameter sweep configuration
    pub fn parameter_sweep(
        base_config: &SimulationConfig,
        parameter: ParameterSweep,
    ) -> Vec<SimulationConfig> {
        match parameter {
            ParameterSweep::Resolution(resolutions) => {
                resolutions.into_iter().map(|res| {
                    let mut config = base_config.clone();
                    config.resolution = res;
                    config.id = format!("{}_res{}", base_config.id, res);
                    config.output_dir = config.output_dir.join(format!("res_{}", res));
                    config
                }).collect()
            }
            ParameterSweep::LaserIntensity(intensities) => {
                intensities.into_iter().map(|intensity| {
                    let mut config = base_config.clone();
                    config.laser.peak_intensity = intensity;
                    config.id = format!("{}_intensity_{:.0e}", base_config.id, intensity);
                    config.output_dir = config.output_dir.join(format!("intensity_{:.0e}", intensity));
                    config
                }).collect()
            }
            ParameterSweep::PerturbationAmplitude(amplitudes) => {
                amplitudes.into_iter().map(|amp| {
                    let mut config = base_config.clone();
                    config.perturbation_amplitude = amp;
                    config.id = format!("{}_amp_{:.1}", base_config.id, amp);
                    config.output_dir = config.output_dir.join(format!("amp_{:.1}", amp));
                    config
                }).collect()
            }
            ParameterSweep::LaserDuration(durations) => {
                durations.into_iter().map(|dur| {
                    let mut config = base_config.clone();
                    config.laser.duration_as = dur;
                    config.id = format!("{}_laser{}as", base_config.id, dur);
                    config.output_dir = config.output_dir.join(format!("laser_{}_as", dur));
                    config
                }).collect()
            }
        }
    }
}

/// Parameter sweep specifications
#[derive(Debug, Clone)]
pub enum ParameterSweep {
    /// Sweep over different grid resolutions
    Resolution(Vec<usize>),

    /// Sweep over laser intensities (W/cm²)
    LaserIntensity(Vec<f64>),

    /// Sweep over perturbation amplitudes (GeV)
    PerturbationAmplitude(Vec<f64>),

    /// Sweep over laser pulse durations (as)
    LaserDuration(Vec<f64>),
}

/// Output from a completed simulation
#[derive(Debug)]
pub struct SimulationOutput {
    /// Configuration used
    pub config: SimulationConfig,

    /// Simulation results
    pub result: SimulationResult,

    /// Elapsed time in seconds
    pub elapsed_seconds: f64,

    /// Performance metric (grid updates per second)
    pub updates_per_second: f64,
}

impl SimulationOutput {
    /// Calculate energy conservation percentage
    pub fn energy_conservation_percent(&self) -> f64 {
        if let (Some(&initial), Some(&final_energy)) = (
            self.result.metrics.total_energy.first(),
            self.result.metrics.total_energy.last(),
        ) {
            ((final_energy - initial).abs() / initial * 100.0)
        } else {
            0.0
        }
    }

    /// Get field value statistics
    pub fn field_statistics(&self) -> FieldStatistics {
        FieldStatistics {
            mean: self.result.final_field.mean_value(),
            min: self.result.final_field.min_value(),
            max: self.result.final_field.max_value(),
            range: self.result.final_field.max_value() - self.result.final_field.min_value(),
        }
    }
}

/// Field value statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
}

/// Comparison report for multiple simulations
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub simulations: Vec<SimulationSummary>,
    pub best_performance: String,
    pub best_conservation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationSummary {
    pub id: String,
    pub resolution: usize,
    pub elapsed_seconds: f64,
    pub updates_per_second: f64,
    pub energy_conservation_percent: f64,
    pub field_stats: FieldStatistics,
}

impl ComparisonReport {
    /// Generate comparison report from multiple outputs
    pub fn from_outputs(outputs: &[SimulationOutput]) -> Self {
        let summaries: Vec<_> = outputs.iter().map(|out| {
            SimulationSummary {
                id: out.config.id.clone(),
                resolution: out.config.resolution,
                elapsed_seconds: out.elapsed_seconds,
                updates_per_second: out.updates_per_second,
                energy_conservation_percent: out.energy_conservation_percent(),
                field_stats: out.field_statistics(),
            }
        }).collect();

        let best_performance = summaries.iter()
            .max_by(|a, b| a.updates_per_second.partial_cmp(&b.updates_per_second).unwrap())
            .map(|s| s.id.clone())
            .unwrap_or_default();

        let best_conservation = summaries.iter()
            .min_by(|a, b| a.energy_conservation_percent.partial_cmp(&b.energy_conservation_percent).unwrap())
            .map(|s| s.id.clone())
            .unwrap_or_default();

        Self {
            simulations: summaries,
            best_performance,
            best_conservation,
        }
    }

    /// Export report as JSON
    pub fn to_json_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Print report to console
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║        Higgs Field Simulation - Comparison Report           ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        for sim in &self.simulations {
            println!("📊 Simulation: {}", sim.id);
            println!("   Resolution: {}³ = {} grid points",
                     sim.resolution, sim.resolution.pow(3));
            println!("   Time: {:.2}s", sim.elapsed_seconds);
            println!("   Performance: {:.2e} updates/sec", sim.updates_per_second);
            println!("   Energy conservation: {:.4}%", sim.energy_conservation_percent);
            println!("   Field range: {:.2} - {:.2} GeV (mean: {:.2})",
                     sim.field_stats.min, sim.field_stats.max, sim.field_stats.mean);
            println!();
        }

        println!("🏆 Best performance: {}", self.best_performance);
        println!("🎯 Best energy conservation: {}", self.best_conservation);
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = SimulationConfig {
            id: "test".to_string(),
            resolution: 32,
            size_nm: 100.0,
            dt_as: 1.0,
            duration_as: 100.0,
            perturbation_amplitude: 5.0,
            perturbation_width: 10.0,
            laser: LaserConfig {
                duration_as: 100.0,
                photon_energy_ev: 50.0,
                peak_intensity: 1e13,
            },
            output_dir: PathBuf::from("test_output"),
        };

        assert_eq!(config.resolution, 32);
        assert_eq!(config.id, "test");
    }

    #[test]
    fn test_parameter_sweep_resolution() {
        let base = SimulationConfig {
            id: "sweep".to_string(),
            resolution: 32,
            size_nm: 100.0,
            dt_as: 1.0,
            duration_as: 100.0,
            perturbation_amplitude: 5.0,
            perturbation_width: 10.0,
            laser: LaserConfig {
                duration_as: 100.0,
                photon_energy_ev: 50.0,
                peak_intensity: 1e13,
            },
            output_dir: PathBuf::from("sweep_output"),
        };

        let configs = SimulationFactory::parameter_sweep(
            &base,
            ParameterSweep::Resolution(vec![16, 32, 64]),
        );

        assert_eq!(configs.len(), 3);
        assert_eq!(configs[0].resolution, 16);
        assert_eq!(configs[1].resolution, 32);
        assert_eq!(configs[2].resolution, 64);
    }
}
