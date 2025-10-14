//! Demonstration of Higgs field simulation with attosecond laser pulse interaction
//!
//! This example shows:
//! 1. Initializing a 3D Higgs field at vacuum expectation value
//! 2. Adding a localized perturbation
//! 3. Applying an attosecond XUV laser pulse
//! 4. Evolving the field and monitoring dynamics
//! 5. Exporting visualization data

use q_higgs_simulator::*;
use anyhow::Result;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("🌊 Higgs Field Simulator - Attosecond Laser Pulse Demo");
    println!("====================================================\n");

    // Simulation parameters
    let resolution = 64;  // 64³ grid points
    let size_nm = 100.0;  // 100 nm simulation box
    let dt_as = 1.0;      // 1 attosecond time step
    let duration_as = 1000.0;  // 1 femtosecond total

    println!("📊 Simulation Parameters:");
    println!("   Resolution: {}³ = {} grid points", resolution, (resolution as u32).pow(3));
    println!("   Box size: {} nm", size_nm);
    println!("   Grid spacing: {:.3} nm", size_nm / resolution as f64);
    println!("   Time step: {} as", dt_as);
    println!("   Duration: {} as ({} fs)\n", duration_as, duration_as / 1000.0);

    // Create simulator
    let mut simulator = HiggsSimulator::new(resolution, size_nm);

    // Initialize field at vacuum expectation value
    println!("⚛️  Initializing Higgs field at VEV = {:.2} GeV",
             simulator.constants.vacuum_expectation_value());
    simulator.initialize_vacuum();

    // Add small Gaussian perturbation
    let perturbation_amplitude = 5.0;  // 5 GeV
    let perturbation_width = 10.0;     // 10 nm
    println!("   Adding Gaussian perturbation:");
    println!("   - Amplitude: {} GeV", perturbation_amplitude);
    println!("   - Width: {} nm\n", perturbation_width);

    simulator.field.add_gaussian_perturbation(perturbation_amplitude, perturbation_width);

    // Create attosecond XUV laser pulse
    let photon_energy_ev = 50.0;  // 50 eV XUV photon
    let pulse_duration_as = 100.0;  // 100 as pulse
    let peak_intensity = 1e13;  // 10¹³ W/cm²

    println!("🔬 Laser Pulse Configuration:");
    println!("   Photon energy: {} eV", photon_energy_ev);
    println!("   Wavelength: {:.1} nm", 1239.84 / photon_energy_ev);
    println!("   Duration (FWHM): {} as", pulse_duration_as);
    println!("   Peak intensity: {:.2e} W/cm²", peak_intensity);

    let laser = LaserPulse::xuv_attosecond(pulse_duration_as, photon_energy_ev, peak_intensity);

    let up_ev = laser.ponderomotive_potential(0.0);
    println!("   Ponderomotive potential: {:.3} eV\n", up_ev);

    simulator.set_laser(laser);

    // Initial field statistics
    let initial_energy = simulator.field.total_energy();
    println!("📈 Initial Field Statistics:");
    println!("   Total energy: {:.6e} GeV", initial_energy);
    println!("   Mean value: {:.2} GeV", simulator.field.mean_value());
    println!("   Max value: {:.2} GeV", simulator.field.max_value());
    println!("   Min value: {:.2} GeV\n", simulator.field.min_value());

    // Run simulation
    println!("🚀 Starting field evolution...");
    let start_time = std::time::Instant::now();

    let result = simulator.simulate(duration_as, dt_as).await?;

    let elapsed = start_time.elapsed();
    println!("   ✅ Simulation complete in {:.2} seconds", elapsed.as_secs_f64());
    println!("   Performance: {:.1} grid-updates/second\n",
             (resolution.pow(3) as f64 * (duration_as / dt_as)) / elapsed.as_secs_f64());

    // Final field statistics
    let final_energy = result.final_field.total_energy();
    println!("📊 Final Field Statistics:");
    println!("   Total energy: {:.6e} GeV", final_energy);
    println!("   Mean value: {:.2} GeV", result.final_field.mean_value());
    println!("   Max value: {:.2} GeV", result.final_field.max_value());
    println!("   Min value: {:.2} GeV", result.final_field.min_value());

    let energy_change_percent = ((final_energy - initial_energy) / initial_energy * 100.0).abs();
    println!("   Energy conservation: {:.2}% change\n", energy_change_percent);

    // Export visualizations
    println!("💾 Exporting visualization data...");

    let output_dir = Path::new("higgs_simulation_output");
    std::fs::create_dir_all(output_dir)?;

    // Export final field state
    VtkExporter::export_scalar_field(
        &result.final_field,
        output_dir.join("higgs_field_final.vtk"),
        "phi",
    )?;
    println!("   ✅ Saved: higgs_field_final.vtk");

    // Export central slice
    let center_z = resolution / 2;
    VtkExporter::export_2d_slice(
        &result.final_field,
        center_z,
        output_dir.join("higgs_slice_xy.vtk"),
        "phi",
    )?;
    println!("   ✅ Saved: higgs_slice_xy.vtk");

    // Export ASCII line profile
    AsciiExporter::export_line_profile(
        &result.final_field,
        output_dir.join("line_profile.txt"),
    )?;
    println!("   ✅ Saved: line_profile.txt");

    // Export metrics
    export_metrics(&result.metrics, output_dir.join("metrics.txt"))?;
    println!("   ✅ Saved: metrics.txt\n");

    // Display metrics summary
    println!("📈 Simulation Metrics:");
    println!("   Total steps recorded: {}", result.metrics.steps.len());

    if let (Some(&min_energy), Some(&max_energy)) = (
        result.metrics.total_energy.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
        result.metrics.total_energy.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
    ) {
        println!("   Energy range: {:.6e} to {:.6e} GeV", min_energy, max_energy);
        println!("   Energy fluctuation: {:.2}%",
                 (max_energy - min_energy) / min_energy * 100.0);
    }

    println!("\n✨ Demonstration complete!");
    println!("\n📁 Output files in: {}", output_dir.display());
    println!("   - Open .vtk files in ParaView for 3D visualization");
    println!("   - Plot line_profile.txt for 1D field profile");
    println!("   - View metrics.txt for detailed statistics");

    Ok(())
}

/// Export metrics to text file
fn export_metrics(metrics: &SimulationMetrics, path: impl AsRef<Path>) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "# Higgs Field Simulation Metrics")?;
    writeln!(file, "# step total_energy max_field min_field")?;

    for i in 0..metrics.steps.len() {
        writeln!(
            file,
            "{} {:.6e} {:.6e} {:.6e}",
            metrics.steps[i],
            metrics.total_energy[i],
            metrics.max_field_value[i],
            metrics.min_field_value[i]
        )?;
    }

    Ok(())
}
