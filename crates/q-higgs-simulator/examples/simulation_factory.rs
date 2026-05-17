//! Higgs Field Simulation Factory Demonstration
//!
//! This example demonstrates industrial-scale parallel simulation orchestration:
//! - Parameter sweeps across multiple dimensions
//! - Parallel execution with concurrency control
//! - Automated comparison and analysis
//! - Performance benchmarking

use q_higgs_simulator::*;
use anyhow::Result;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🏭 Higgs Field Simulation Factory - Industrial Scale Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create factory with 4 concurrent simulations
    let factory = SimulationFactory::new(4);

    println!("🔧 Factory Configuration:");
    println!("   Max concurrent simulations: 4");
    println!("   Target: Parameter sweep across resolutions\n");

    // Create base configuration
    let base_config = SimulationConfig {
        id: "higgs_sweep".to_string(),
        resolution: 32, // Will be overridden by sweep
        size_nm: 100.0,
        dt_as: 1.0,
        duration_as: 500.0, // Shorter for demo (0.5 femtoseconds)
        perturbation_amplitude: 5.0,
        perturbation_width: 10.0,
        laser: LaserConfig {
            duration_as: 100.0,
            photon_energy_ev: 50.0,
            peak_intensity: 1e13,
        },
        output_dir: PathBuf::from("factory_output"),
    };

    println!("📊 Base Configuration:");
    println!("   Box size: {} nm", base_config.size_nm);
    println!("   Duration: {} as ({} fs)", base_config.duration_as, base_config.duration_as / 1000.0);
    println!("   Time step: {} as", base_config.dt_as);
    println!("   Perturbation: {} GeV amplitude, {} nm width",
             base_config.perturbation_amplitude, base_config.perturbation_width);
    println!("   Laser: {} eV, {} as pulse, {:.2e} W/cm²\n",
             base_config.laser.photon_energy_ev,
             base_config.laser.duration_as,
             base_config.laser.peak_intensity);

    // Generate parameter sweep across resolutions
    println!("🔬 Parameter Sweep: Grid Resolution");
    println!("   Testing: 16³, 32³, 48³ grid points\n");

    let configs = SimulationFactory::parameter_sweep(
        &base_config,
        ParameterSweep::Resolution(vec![16, 32, 48]),
    );

    println!("📋 Generated {} simulation configurations:", configs.len());
    for config in &configs {
        println!("   • {} - {}³ = {} grid points",
                 config.id, config.resolution, config.resolution.pow(3));
    }
    println!();

    // Run batch simulation
    println!("🚀 Starting parallel batch execution...\n");
    let start_time = std::time::Instant::now();

    let outputs = factory.run_batch(configs).await?;

    let total_elapsed = start_time.elapsed();
    println!("✅ Batch complete in {:.2}s\n", total_elapsed.as_secs_f64());

    // Generate comparison report
    println!("📈 Generating comparison report...\n");
    let report = ComparisonReport::from_outputs(&outputs);

    // Print report to console
    report.print();

    // Export report to JSON
    let report_path = PathBuf::from("factory_output/comparison_report.json");
    std::fs::create_dir_all("factory_output")?;
    report.to_json_file(&report_path)?;
    println!("💾 Report saved to: {}\n", report_path.display());

    // Additional analysis
    println!("📊 Detailed Analysis:");
    println!("───────────────────────────────────────────────────────────");

    for output in &outputs {
        let grid_points = output.config.resolution.pow(3);
        let timesteps = (output.config.duration_as / output.config.dt_as) as usize;
        let total_ops = grid_points * timesteps;

        println!("\n🔍 {}:", output.config.id);
        println!("   Grid: {}³ = {} points", output.config.resolution, grid_points);
        println!("   Timesteps: {}", timesteps);
        println!("   Total operations: {:.2e}", total_ops as f64);
        println!("   Elapsed time: {:.2}s", output.elapsed_seconds);
        println!("   Performance: {:.2e} updates/sec", output.updates_per_second);
        println!("   Throughput: {:.2} Mops/sec", (total_ops as f64 / output.elapsed_seconds) / 1e6);

        let stats = output.field_statistics();
        println!("   Final field: mean={:.2} GeV, range={:.2} GeV",
                 stats.mean, stats.range);
        println!("   Energy conservation: {:.4}%", output.energy_conservation_percent());
    }

    println!("\n───────────────────────────────────────────────────────────");

    // Scaling analysis
    if outputs.len() >= 2 {
        println!("\n🔬 Scaling Analysis:");
        let base_output = &outputs[0];

        for output in &outputs[1..] {
            let grid_ratio = (output.config.resolution as f64 / base_output.config.resolution as f64).powi(3);
            let time_ratio = output.elapsed_seconds / base_output.elapsed_seconds;
            let scaling_efficiency = (grid_ratio / time_ratio) * 100.0;

            println!("   {}³ vs {}³:", output.config.resolution, base_output.config.resolution);
            println!("      Grid points ratio: {:.2}x", grid_ratio);
            println!("      Time ratio: {:.2}x", time_ratio);
            println!("      Scaling efficiency: {:.1}%", scaling_efficiency);
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  🏭 Simulation Factory Demo Complete                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\n📁 All output files saved to: factory_output/");
    println!("📊 View VTK files in ParaView for 3D visualization");
    println!("📈 JSON report available for further analysis\n");

    // Example: Laser intensity sweep
    println!("💡 Additional Factory Capabilities:\n");
    println!("   🔸 Laser Intensity Sweep:");
    let intensity_configs = SimulationFactory::parameter_sweep(
        &base_config,
        ParameterSweep::LaserIntensity(vec![1e12, 1e13, 1e14]),
    );
    println!("      Generated {} configurations", intensity_configs.len());

    println!("\n   🔸 Perturbation Amplitude Sweep:");
    let amplitude_configs = SimulationFactory::parameter_sweep(
        &base_config,
        ParameterSweep::PerturbationAmplitude(vec![1.0, 5.0, 10.0]),
    );
    println!("      Generated {} configurations", amplitude_configs.len());

    println!("\n   🔸 Laser Duration Sweep:");
    let duration_configs = SimulationFactory::parameter_sweep(
        &base_config,
        ParameterSweep::LaserDuration(vec![50.0, 100.0, 200.0]),
    );
    println!("      Generated {} configurations", duration_configs.len());

    println!("\n🚀 Factory ready for production-scale quantum field simulations!");

    Ok(())
}
