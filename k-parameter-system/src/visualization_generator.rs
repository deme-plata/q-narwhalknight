//! Comprehensive Visualization Generator for K-Parameter Quantum Frontiers Paper
//! Generates all figures referenced in the LaTeX document

use k_biological_quantum::{calculate_k_bio, BiologicalQuantumSystem};
use k_constants::{PhysicalConstants, PlanckScales};
use k_cosmological_inflation::{SlowRollParameters, StarobinskyPotential, PowerSpectrum};
use k_dark_sector::{calculate_k_dark, DarkEnergy, DarkMatter, nfw_density};
use k_foam_topology::QuantumFoamTopology;
use k_graph_generator::{DataSeries, GraphPlotter};
use k_quantum_gravity::{calculate_k_gravity, BlackHoleEvaporation};
use k_topological_quantum::{AnyonType, BerryPhase, TopologicalState};

use std::f64::consts::PI;

mod landscape_generators;
use landscape_generators::{
    generate_variant_a_core_landscape,
    generate_variant_b_gravitational,
    generate_variant_c_dark_sector,
    generate_all_landscapes,
};

mod enhanced_figure_generators;
use enhanced_figure_generators::{
    generate_enhanced_fig2_gravitational,
    generate_enhanced_fig3_dark_matter,
};

/// Generate Figure 1: K-Parameter Evolution Across All Quantum Sectors
pub fn generate_fig1_k_parameter_overview() -> Result<(), Box<dyn std::error::Error>> {
    let constants = PhysicalConstants::default();
    let times: Vec<f64> = (0..200).map(|i| i as f64 * 1e11).collect(); // 0 to 2e13 seconds
    let k_standard = 1e15;

    // Gravitational K
    let k_gravity: Vec<f64> = times.iter().map(|&t| {
        let mass = 1.989e30; // Solar mass
        let radius = 1e10 + t * 1e-6; // Slowly increasing radius
        calculate_k_gravity(k_standard, mass, radius, 1.0, &constants)
    }).collect();

    // Dark sector K
    let dm = DarkMatter::default();
    let de = DarkEnergy::default();
    let k_dark: Vec<f64> = times.iter().map(|&t| {
        calculate_k_dark(k_standard, &dm, &de, t)
    }).collect();

    // Biological K
    let bio_system = BiologicalQuantumSystem::default();
    let k_bio: Vec<f64> = times.iter().map(|&t| {
        calculate_k_bio(k_standard, &bio_system, t * 1e-12) // Picosecond scale for bio
    }).collect();

    // Topological K (magnitude only)
    let topo_state = TopologicalState::new(AnyonType::Fibonacci, 4);
    let k_topo: Vec<f64> = times.iter().map(|&t| {
        let n = (t / 1e12) as i32;
        k_standard * topo_state.golden_ratio.powi(2 * (n % 10))
    }).collect();

    let times_years: Vec<f64> = times.iter().map(|&t| t / (365.25 * 86400.0)).collect();

    let series = vec![
        DataSeries::new("Standard K", times_years.clone(), vec![k_standard; times.len()]).with_color("black"),
        DataSeries::new("Gravitational K", times_years.clone(), k_gravity).with_color("red"),
        DataSeries::new("Dark Sector K", times_years.clone(), k_dark).with_color("blue"),
        DataSeries::new("Biological K", times_years.clone(), k_bio).with_color("green"),
        DataSeries::new("Topological K", times_years.clone(), k_topo).with_color("magenta"),
    ];

    let plotter = GraphPlotter::new(
        "K-Parameter Evolution Across Quantum Sectors",
        "Time (years)",
        "K-Parameter Value",
    );

    plotter.plot_to_file(&series, "figures/fig1_k_parameter_overview.png")?;
    println!("Generated: fig1_k_parameter_overview.png");

    Ok(())
}

/// Generate Figure 2: Gravitational K-Parameter vs Distance from Mass
pub fn generate_fig2_gravitational_k() -> Result<(), Box<dyn std::error::Error>> {
    let constants = PhysicalConstants::default();
    let k_standard = 1e15;
    let mass = 1.989e30; // Solar mass

    let radii: Vec<f64> = (10..200).map(|i| 10f64.powi(i as i32 / 20) * 1e6).collect(); // Log scale from 1e6 to 1e16 m

    let k_gravity: Vec<f64> = radii.iter().map(|&r| {
        calculate_k_gravity(k_standard, mass, r, 1.0, &constants)
    }).collect();

    let radii_km: Vec<f64> = radii.iter().map(|&r| r / 1e3).collect();

    let series = vec![
        DataSeries::new("K_gravity", radii_km.clone(), k_gravity).with_color("red"),
        DataSeries::new("K_standard (baseline)", radii_km.clone(), vec![k_standard; radii.len()]).with_color("black"),
    ];

    let plotter = GraphPlotter::new(
        "Gravitational K-Parameter Enhancement vs Distance",
        "Distance from Solar Mass (km)",
        "K-Parameter Value",
    );

    plotter.plot_to_file(&series, "figures/fig2_gravitational_k.png")?;
    println!("Generated: fig2_gravitational_k.png");

    Ok(())
}

/// Generate Figure 3: Dark Matter Annual Modulation
pub fn generate_fig3_dark_matter_modulation() -> Result<(), Box<dyn std::error::Error>> {
    let dm = DarkMatter::default();
    let de = DarkEnergy::default();
    let k_standard = 1e15;

    let days: Vec<f64> = (0..365).map(|d| d as f64).collect();
    let times: Vec<f64> = days.iter().map(|&d| d * 86400.0).collect(); // Convert to seconds

    // Annual modulation: amplitude 7%, peak in June (day 150)
    let modulation: Vec<f64> = times.iter().enumerate().map(|(i, &t)| {
        let base = calculate_k_dark(k_standard, &dm, &de, t);
        let annual_factor = 1.0 + 0.07 * (2.0 * PI * (days[i] - 150.0) / 365.0).cos();
        base * annual_factor
    }).collect();

    let series = vec![
        DataSeries::new("K_dark with Annual Modulation", days.clone(), modulation).with_color("blue"),
        DataSeries::new("K_standard", days.clone(), vec![k_standard; days.len()]).with_color("black"),
    ];

    let plotter = GraphPlotter::new(
        "Dark Matter K-Parameter Annual Modulation",
        "Day of Year",
        "K-Parameter Value",
    );

    plotter.plot_to_file(&series, "figures/fig3_dark_matter_modulation.png")?;
    println!("Generated: fig3_dark_matter_modulation.png");

    Ok(())
}

/// Generate Figure 4: Biological Coherence Time Evolution
pub fn generate_fig4_biological_coherence() -> Result<(), Box<dyn std::error::Error>> {
    let k_quantum = 1e15;

    let times_ps: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect(); // 0-10 ps
    let times: Vec<f64> = times_ps.iter().map(|&t| t * 1e-12).collect(); // Convert to seconds

    // FMO complex (Φ_bio = 0.95, Γ = 1.5e12 Hz)
    let bio_fmo = BiologicalQuantumSystem { phi_bio: 0.95, gamma_dephasing: 1.5e12, temperature: 300.0 };
    let k_fmo: Vec<f64> = times.iter().map(|&t| calculate_k_bio(k_quantum, &bio_fmo, t)).collect();

    // Cryptophyte (Φ_bio = 0.85, Γ = 1e12 Hz)
    let bio_crypto = BiologicalQuantumSystem { phi_bio: 0.85, gamma_dephasing: 1e12, temperature: 300.0 };
    let k_crypto: Vec<f64> = times.iter().map(|&t| calculate_k_bio(k_quantum, &bio_crypto, t)).collect();

    // Neural microtubule (Φ_bio = 0.1, Γ = 4e10 Hz)
    let bio_neural = BiologicalQuantumSystem { phi_bio: 0.1, gamma_dephasing: 4e10, temperature: 310.0 };
    let k_neural: Vec<f64> = times.iter().map(|&t| calculate_k_bio(k_quantum, &bio_neural, t)).collect();

    let series = vec![
        DataSeries::new("FMO Complex (Φ=0.95)", times_ps.clone(), k_fmo).with_color("green"),
        DataSeries::new("Cryptophyte (Φ=0.85)", times_ps.clone(), k_crypto).with_color("cyan"),
        DataSeries::new("Neural Microtubule (Φ=0.1)", times_ps.clone(), k_neural).with_color("magenta"),
    ];

    let plotter = GraphPlotter::new(
        "Biological Quantum Coherence Decay",
        "Time (picoseconds)",
        "K_bio Parameter",
    );

    plotter.plot_to_file(&series, "figures/fig4_biological_coherence.png")?;
    println!("Generated: fig4_biological_coherence.png");

    Ok(())
}

/// Generate Figure 5: Topological Quantum Dimension Scaling
pub fn generate_fig5_topological_scaling() -> Result<(), Box<dyn std::error::Error>> {
    let n_anyons: Vec<f64> = (1..20).map(|n| n as f64).collect();

    // Fibonacci anyons (golden ratio scaling)
    let fib_state = TopologicalState::new(AnyonType::Fibonacci, 1);
    let fib_dims: Vec<f64> = n_anyons.iter().map(|&n| {
        fib_state.golden_ratio.powf(n)
    }).collect();

    // Ising anyons (sqrt(2) scaling)
    let ising_state = TopologicalState::new(AnyonType::Ising, 1);
    let ising_dims: Vec<f64> = n_anyons.iter().map(|&n| {
        ising_state.quantum_dimension().powf(n)
    }).collect();

    // Conventional qubits (2^n scaling)
    let qubit_dims: Vec<f64> = n_anyons.iter().map(|&n| 2.0_f64.powf(n)).collect();

    let series = vec![
        DataSeries::new("Fibonacci Anyons (φ^n)", n_anyons.clone(), fib_dims).with_color("magenta"),
        DataSeries::new("Ising Anyons (√2^n)", n_anyons.clone(), ising_dims).with_color("cyan"),
        DataSeries::new("Conventional Qubits (2^n)", n_anyons.clone(), qubit_dims).with_color("black"),
    ];

    let plotter = GraphPlotter::new(
        "Topological Quantum Dimension Scaling",
        "Number of Particles (n)",
        "Quantum Dimension",
    );

    plotter.plot_to_file(&series, "figures/fig5_topological_scaling.png")?;
    println!("Generated: fig5_topological_scaling.png");

    Ok(())
}

/// Generate Figure 6: Cosmological Inflation Power Spectrum
pub fn generate_fig6_power_spectrum() -> Result<(), Box<dyn std::error::Error>> {
    let constants = PhysicalConstants::default();
    let potential = StarobinskyPotential::new(1e16, &constants);
    let phi = 3.0 * potential.m_planck;

    let v = potential.potential(phi);
    let sr = SlowRollParameters::from_potential(&potential, phi);
    let ps = PowerSpectrum::from_slow_roll(&sr, v, potential.m_planck);

    let k_pivot = 0.05; // Mpc^-1
    let k_values: Vec<f64> = (0..100).map(|i| k_pivot * 10f64.powf((i as f64 - 50.0) / 25.0)).collect();

    let power: Vec<f64> = k_values.iter().map(|&k| ps.evaluate(k, k_pivot)).collect();

    let series = vec![
        DataSeries::new(&format!("Power Spectrum (n_s={:.4})", ps.spectral_index),
                       k_values.clone(), power).with_color("red"),
    ];

    let plotter = GraphPlotter::new(
        "Primordial Power Spectrum from Starobinsky Inflation",
        "Wavenumber k (Mpc^-1)",
        "Power P_R(k)",
    );

    plotter.plot_to_file(&series, "figures/fig6_power_spectrum.png")?;
    println!("Generated: fig6_power_spectrum.png");

    Ok(())
}

/// Generate Figure 7: Quantum Foam Topology Network
pub fn generate_fig7_quantum_foam() -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    use plotters::element::{Circle, PathElement};
    use plotters::series::LineSeries;

    let constants = PhysicalConstants::default();
    let planck = PlanckScales::from_constants(&constants);
    let mut foam = QuantumFoamTopology::new(50, &planck);
    foam.generate_random_connections(0.2);

    // Create custom plot to show both edges and nodes
    let root = BitMapBackend::new("figures/fig7_quantum_foam.png", (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Find plot bounds from node positions
    let x_coords: Vec<f64> = foam.nodes.iter().map(|n| n.position[0]).collect();
    let y_coords: Vec<f64> = foam.nodes.iter().map(|n| n.position[1]).collect();

    let x_min = x_coords.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_coords.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let margin = 1.0;
    let mut chart = ChartBuilder::on(&root)
        .caption("Quantum Foam Topology Network Structure", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (x_min - margin)..(x_max + margin),
            (y_min - margin)..(y_max + margin)
        )?;

    chart.configure_mesh()
        .x_desc("X Position (Planck units)")
        .y_desc("Y Position (Planck units)")
        .draw()?;

    // Draw edges FIRST (underneath nodes)
    for edge in &foam.edges {
        let node1 = &foam.nodes[edge.source];
        let node2 = &foam.nodes[edge.target];

        let edge_color = match edge.causality {
            -1 => &BLUE.mix(0.3),   // Timelike - blue
            0 => &GREEN.mix(0.3),    // Null - green
            1 => &RED.mix(0.3),      // Spacelike - red
            _ => &BLACK.mix(0.3),
        };

        chart.draw_series(LineSeries::new(
            vec![
                (node1.position[0], node1.position[1]),
                (node2.position[0], node2.position[1])
            ],
            edge_color,
        ))?;
    }

    // Draw nodes on top as circles
    chart.draw_series(
        foam.nodes.iter().map(|node| {
            Circle::new(
                (node.position[0], node.position[1]),
                4,
                BLACK.filled()
            )
        })
    )?
    .label(&format!("{} nodes, {} edges", foam.nodes.len(), foam.edges.len()))
    .legend(|(x, y)| Circle::new((x + 10, y), 3, BLACK.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: fig7_quantum_foam.png (with {} nodes, {} edges)",
             foam.nodes.len(), foam.edges.len());

    Ok(())
}

/// Generate Figure 8: Black Hole Evaporation Timeline
pub fn generate_fig8_hawking_evaporation() -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    use plotters::coord::types::RangedCoordf64;

    let constants = PhysicalConstants::default();

    // Use smaller initial masses for visible evaporation
    let masses_kg = vec![1e8, 1e10, 1e12]; // Much smaller - evaporate in hours to years
    let mass_labels = vec!["Small", "Medium", "Large"];
    let colors = [&RED, &BLUE, &GREEN];

    let root = BitMapBackend::new("figures/fig8_hawking_evaporation.png", (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Collect all data for range calculation
    let mut all_data = Vec::new();
    for &initial_mass in &masses_kg {
        let bh = BlackHoleEvaporation::new(initial_mass);
        let t_total = bh.total_evaporation_time(&constants);

        let times: Vec<f64> = (1..100).map(|j| j as f64 * t_total / 100.0).collect();
        let masses: Vec<f64> = times.iter().map(|&t| {
            let mut bh_copy = bh.clone();
            let dt = t / 1000.0;
            for _ in 0..1000 {
                bh_copy.evolve(dt, &constants);
            }
            bh_copy.current_mass.max(1.0) // Avoid zero for log scale
        }).collect();

        let times_years: Vec<f64> = times.iter().map(|&t| t / (365.25 * 86400.0)).collect();
        all_data.push((times_years, masses));
    }

    // Build log-log chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Black Hole Hawking Evaporation (Log-Log Scale)", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (1e-10_f64..1e25_f64).log_scale(),
            (1e0_f64..1e13_f64).log_scale()
        )?;

    chart.configure_mesh()
        .x_desc("Time (years)")
        .y_desc("Black Hole Mass (kg)")
        .draw()?;

    // Plot each series
    for (i, (times, masses)) in all_data.iter().enumerate() {
        let points: Vec<(f64, f64)> = times.iter().zip(masses.iter())
            .map(|(&t, &m)| (t, m))
            .collect();

        chart.draw_series(LineSeries::new(points, colors[i]))?
            .label(&format!("{} ({:.0e} kg)", mass_labels[i], masses_kg[i]))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Generated: fig8_hawking_evaporation.png (log-log scale)");

    Ok(())
}

/// Generate Figure 9: NFW Dark Matter Halo Profile
pub fn generate_fig9_nfw_profile() -> Result<(), Box<dyn std::error::Error>> {
    let scale_radius = 20e3 * 3.086e19; // 20 kpc in meters
    let rho_s = 0.2 * 1.783e-27; // 0.2 GeV/cm³ in kg/m³

    let radii_kpc: Vec<f64> = (1..100).map(|i| i as f64 * 0.5).collect();
    let radii: Vec<f64> = radii_kpc.iter().map(|&r| r * 1e3 * 3.086e19).collect(); // Convert to meters

    let densities: Vec<f64> = radii.iter().map(|&r| {
        nfw_density(r, scale_radius, rho_s) / 1.783e-27 // Convert back to GeV/cm³
    }).collect();

    let series = vec![
        DataSeries::new("NFW Profile (r_s=20 kpc)", radii_kpc, densities).with_color("blue"),
    ];

    let plotter = GraphPlotter::new(
        "Dark Matter Halo Density Profile (NFW)",
        "Radius (kpc)",
        "Density (GeV/cm³)",
    );

    plotter.plot_to_file(&series, "figures/fig9_nfw_profile.png")?;
    println!("Generated: fig9_nfw_profile.png");

    Ok(())
}

/// Generate Figure 10: Berry Phase Evolution
pub fn generate_fig10_berry_phase() -> Result<(), Box<dyn std::error::Error>> {
    let n_steps_values = vec![10, 50, 100, 500];

    let mut all_series = Vec::new();

    for (i, &n_steps) in n_steps_values.iter().enumerate() {
        let mut berry = BerryPhase::new();
        let steps: Vec<f64> = (0..n_steps).map(|j| j as f64).collect();
        let phases: Vec<f64> = steps.iter().map(|&s| {
            berry.path_parameter = 2.0 * PI * s / n_steps as f64;
            berry.calculate_geometric_phase(s as usize + 1)
        }).collect();

        let colors = ["red", "blue", "green", "magenta"];
        all_series.push(
            DataSeries::new(&format!("n={} steps", n_steps), steps, phases)
                .with_color(colors[i])
        );
    }

    let plotter = GraphPlotter::new(
        "Berry Phase Accumulation During Adiabatic Evolution",
        "Evolution Step",
        "Berry Phase (radians)",
    );

    plotter.plot_to_file(&all_series, "figures/fig10_berry_phase.png")?;
    println!("Generated: fig10_berry_phase.png");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  K-Parameter Quantum Frontiers Visualization Generator   ║");
    println!("║  Creating Comprehensive Figures for LaTeX Document       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Create figures directory
    std::fs::create_dir_all("figures")?;

    // Generate all figures
    println!("Generating Figure 1: K-Parameter Overview...");
    generate_fig1_k_parameter_overview()?;

    println!("\nGenerating Figure 2: Gravitational K-Parameter...");
    generate_fig2_gravitational_k()?;

    println!("\nGenerating Figure 3: Dark Matter Modulation...");
    generate_fig3_dark_matter_modulation()?;

    println!("\nGenerating Figure 4: Biological Coherence...");
    generate_fig4_biological_coherence()?;

    println!("\nGenerating Figure 5: Topological Scaling...");
    generate_fig5_topological_scaling()?;

    println!("\nGenerating Figure 6: Power Spectrum...");
    generate_fig6_power_spectrum()?;

    println!("\nGenerating Figure 7: Quantum Foam...");
    generate_fig7_quantum_foam()?;

    println!("\nGenerating Figure 8: Hawking Evaporation...");
    generate_fig8_hawking_evaporation()?;

    println!("\nGenerating Figure 9: NFW Halo Profile...");
    generate_fig9_nfw_profile()?;

    println!("\nGenerating Figure 10: Berry Phase...");
    generate_fig10_berry_phase()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Generating K-Parameter Landscape Figures (NEW!)          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    generate_all_landscapes()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Generating Enhanced Publication-Quality Figures          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Generating Enhanced Figure 2: Gravitational (log scale + bands)...");
    generate_enhanced_fig2_gravitational()?;

    println!("\nGenerating Enhanced Figure 3: Dark Matter (multi-panel)...");
    generate_enhanced_fig3_dark_matter()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  All Figures Generated Successfully!                      ║");
    println!("║  Output: k-parameter-system/figures/*.png                ║");
    println!("║  - 10 standard figures                                    ║");
    println!("║  - 3 landscape heatmaps (Variants A, B, C)               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
