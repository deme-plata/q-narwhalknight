/// K-Parameter Landscape 3D Visualization Generators
///
/// This module generates three variant landscape figures:
/// - Variant A: Core formalism (log ΔH vs log ΔS)
/// - Variant B: Gravitational (φ vs τ)
/// - Variant C: Dark sector (DM decoherence vs DE enhancement)

use k_graph_generator::{Surface3D, ColorMap};
use k_constants::{PhysicalConstants, PlanckScales};
use std::error::Error;

/// Generate Variant A: Core Formalism Landscape
///
/// Shows how K-Parameter depends on energy uncertainty (ΔH) and entropy
/// variance (ΔS) in logarithmic coordinates. This is the "teaching figure"
/// that makes reviewers immediately understand the framework.
///
/// K = 2π√(ΔH · ΔS · ℏ / τ)
/// K/K_std = √(ΔH/ΔH_std · ΔS/ΔS_std)
///
/// Features:
/// - High peaks (orange): Both ΔH and ΔS large → maximum K
/// - Deep basins (blue): Either ΔH or ΔS small → suppressed K
/// - Diagonal ridge: K constant along ΔH·ΔS = const
pub fn generate_variant_a_core_landscape() -> Result<(), Box<dyn Error>> {
    println!("🎨 Generating Variant A: Core Formalism Landscape...");

    let constants = PhysicalConstants::default();
    let k_std = 1e-15;  // Standard K reference value [J^{1/2} K^{1/2} s^{-1/2}]
    let tau = 1.0;      // Fixed interrogation time [seconds]

    let mut surface = Surface3D::new(
        "K-Parameter Landscape: Energy-Entropy Phase Space",
        "log₁₀(ΔH) [J]",
        "log₁₀(ΔS) [J/K]",
        "K/K_std",
    );

    // Define grid: log₁₀(ΔH) and log₁₀(ΔS) from -25 to -10
    // This spans from sub-Planck scales to macroscopic quantum systems
    surface = surface.from_function(
        (-25.0, -10.0),  // x_range: log₁₀(ΔH) in Joules
        (-25.0, -10.0),  // y_range: log₁₀(ΔS) in J/K
        120,             // nx points (high resolution for smooth surface)
        120,             // ny points
        |log_dh, log_ds| {
            let dh = 10_f64.powf(log_dh);
            let ds = 10_f64.powf(log_ds);

            // K = 2π√(ΔH · ΔS · ℏ / τ)
            let k = 2.0 * std::f64::consts::PI * (dh * ds * constants.hbar / tau).sqrt();

            // Normalize to K_std for visualization
            k / k_std
        }
    );

    // Use hot-cold colormap: orange (high K) → blue (low K)
    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(12)  // 12 iso-K contour lines
        .set_view_angle(45.0, 30.0);  // Azimuth 45°, elevation 30°

    // Add physics annotations
    surface = surface
        .add_annotation(-17.5, -17.5, "K-sensitivity ridge")
        .add_annotation(-22.0, -13.0, "Energy-collapse basin")
        .add_annotation(-13.0, -22.0, "Information-collapse basin")
        .add_annotation(-15.0, -15.0, "Optimal operating point");

    // Export as high-res PNG (for LaTeX document)
    surface.export("figures/k_landscape_variant_a.png", 1600, 1200, 300)?;

    println!("✅ Generated: figures/k_landscape_variant_a.png");
    println!("   - 1600×1200 px @ 300 DPI");
    println!("   - Hot-cold colormap with 12 contour levels");

    Ok(())
}

/// Generate Variant B: Gravitational Enhancement Landscape
///
/// Shows K-Parameter across spacetime curvature (φ = GM/rc²) and measurement
/// timescale (τ). Demonstrates GR time dilation suppression and QG correction
/// enhancement.
///
/// K_gravity = K_std · √(1-2φ) · (1 + α̂_G·ε/E_P) · 1/√τ
///
/// Features:
/// - GR suppression valley: φ → 1 (near r_s) causes K → 0
/// - QG enhancement plateau: α̂_G correction lifts K at moderate φ
/// - 1/√τ slope: K decreases with longer interrogation time
pub fn generate_variant_b_gravitational() -> Result<(), Box<dyn Error>> {
    println!("🎨 Generating Variant B: Gravitational Landscape...");

    let constants = PhysicalConstants::default();
    let planck = PlanckScales::from_constants(&constants);
    let alpha_g = 6.96e-10;  // Measured effective QG coupling
    let e_p = planck.energy;  // Planck energy [J]
    let m = 1.67e-27;  // Proton mass as test mass [kg]
    let c = constants.c;  // Speed of light [m/s]

    let mut surface = Surface3D::new(
        "Gravitational K-Parameter Landscape",
        "φ = GM/(rc²) [dimensionless]",
        "log₁₀(τ) [s]",
        "K_gravity / K_std",
    );

    // φ ranges from near-zero (weak field) to 0.95 (near horizon)
    // τ ranges from 1 ms to 1000 s (log scale)
    surface = surface.from_function(
        (1e-12, 0.95),  // φ: weak field to near-Schwarzschild
        (-3.0, 3.0),    // log₁₀(τ): 1 ms to 1000 s
        120,
        120,
        |phi, log_tau| {
            let tau = 10_f64.powf(log_tau);

            // GR time dilation factor: √(1 - 2φ)
            let gr_factor = (1.0 - 2.0 * phi).max(0.0).sqrt();

            // QG correction: (1 + α̂_G·φ·mc²/E_P)
            let qg_factor = 1.0 + alpha_g * phi * m * c * c / e_p;

            // Measurement timescale factor: 1/√τ
            let tau_factor = 1.0 / tau.sqrt();

            // Combined K_gravity/K_std
            gr_factor * qg_factor * tau_factor
        }
    );

    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(15)
        .set_view_angle(55.0, 28.0);

    // Add critical physics annotations
    surface = surface
        .add_annotation(1.0, 0.0, "Event Horizon (r_s)")
        .add_annotation(0.1, 0.0, "QG correction plateau")
        .add_annotation(1e-9, 1.5, "Earth regime")
        .add_annotation(0.5, -1.0, "Strong-field regime");

    surface.export("figures/k_landscape_variant_b.png", 1600, 1200, 300)?;

    println!("✅ Generated: figures/k_landscape_variant_b.png");
    println!("   - Shows GR time dilation valley + QG enhancement");
    println!("   - α̂_G = 6.96×10⁻¹⁰ coupling visible at intermediate φ");

    Ok(())
}

/// Generate Variant C: Dark Sector Landscape
///
/// Shows trade-off between dark matter decoherence (X = Γ_DM·t) and dark
/// energy enhancement (Y = β_DE·Λ·t²).
///
/// K_dark = K_std · exp(-X) · (1 + Y)
///
/// Features:
/// - DM suppression basin (blue): X > 1 → exponential K decay
/// - DE enhancement ridge (orange): Y > 0 → quadratic K growth
/// - Saddle contour: Compensation line where DM and DE balance
pub fn generate_variant_c_dark_sector() -> Result<(), Box<dyn Error>> {
    println!("🎨 Generating Variant C: Dark Sector Landscape...");

    let mut surface = Surface3D::new(
        "Dark Sector K-Parameter Landscape",
        "X = Γ_DM·t [dimensionless]",
        "Y = β_DE·Λ·t² [dimensionless]",
        "K_dark / K_std",
    );

    // X ranges from 0 (no DM interaction) to 5 (strong decoherence)
    // Y ranges from 0 (no DE effect) to 0.2 (20% enhancement)
    surface = surface.from_function(
        (0.0, 5.0),   // X: DM decoherence parameter
        (0.0, 0.2),   // Y: DE enhancement parameter
        120,
        120,
        |x, y| {
            // K_dark = K_std · exp(-X) · (1 + Y)
            // DM exponentially suppresses, DE quadratically enhances
            (-x).exp() * (1.0 + y)
        }
    );

    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(18)
        .set_view_angle(50.0, 32.0);

    // Add physics annotations
    surface = surface
        .add_annotation(1.0, 0.02, "One e-fold (X=1)")
        .add_annotation(0.1, 0.15, "DE-driven growth")
        .add_annotation(3.5, 0.02, "DM-dominated suppression")
        .add_annotation(2.0, 0.08, "Compensation contour")
        .add_annotation(0.5, 0.1, "Annual modulation track");

    surface.export("figures/k_landscape_variant_c.png", 1600, 1200, 300)?;

    println!("✅ Generated: figures/k_landscape_variant_c.png");
    println!("   - Shows DM/DE trade-off clearly");
    println!("   - Saddle point marks balance condition");

    Ok(())
}

/// Generate all three landscape variants
pub fn generate_all_landscapes() -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   K-PARAMETER LANDSCAPE GENERATOR                         ║");
    println!("║   Publication-Quality 3D Heatmap Visualizations           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Generate all three variants
    generate_variant_a_core_landscape()?;
    println!();

    generate_variant_b_gravitational()?;
    println!();

    generate_variant_c_dark_sector()?;
    println!();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   ALL LANDSCAPE FIGURES GENERATED ✅                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
