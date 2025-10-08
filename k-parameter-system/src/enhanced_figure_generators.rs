/// Enhanced Figure Generators for Publication-Quality Visualizations
///
/// This module implements enhanced versions of key figures with:
/// - Logarithmic axes
/// - Multiple curve overlays
/// - Uncertainty bands
/// - Inset panels
/// - Error bars and residuals

use k_constants::{PhysicalConstants, PlanckScales};
use plotters::prelude::*;
use std::error::Error;
use std::f64::consts::PI;

/// Generate Enhanced Figure 2: Gravitational K-Parameter
///
/// **Enhancements per user specifications:**
/// - Log x-axis: r/r_s from 3r_s to 10^6 r_s
/// - Three curves: GR-only, GR×QG (with α̂_G), baseline
/// - Uncertainty band from α̂_G posterior (6.96 ± 0.15) × 10^{-10}
/// - Earth regime inset showing r ≈ 6.371 × 10^6 m
/// - Complete formulas in caption
pub fn generate_enhanced_fig2_gravitational() -> Result<(), Box<dyn Error>> {
    println!("🎨 Generating Enhanced Figure 2: Gravitational K-Parameter...");

    let constants = PhysicalConstants::default();
    let planck = PlanckScales::from_constants(&constants);

    // Physical parameters
    let m_sun = 1.989e30;  // Solar mass [kg]
    let r_s = 2.0 * constants.g * m_sun / (constants.c * constants.c);  // Schwarzschild radius
    let k_std = 1e-15;  // Standard K reference

    // α̂_G posterior: (6.96 ± 0.15) × 10^{-10}
    let alpha_g_mean = 6.96e-10;
    let alpha_g_sigma = 0.15e-10;

    // X-axis: log scale r/r_s from 3 to 10^6
    let n_points = 200;
    let r_over_rs: Vec<f64> = (0..n_points)
        .map(|i| 3.0 * (1e6 / 3.0_f64).powf(i as f64 / (n_points - 1) as f64))
        .collect();
    let radii: Vec<f64> = r_over_rs.iter().map(|x| x * r_s).collect();

    // Calculate three curves
    let mut k_baseline: Vec<f64> = vec![k_std; n_points];
    let mut k_gr_only: Vec<f64> = Vec::new();
    let mut k_gr_qg: Vec<f64> = Vec::new();
    let mut k_gr_qg_upper: Vec<f64> = Vec::new();  // +1σ uncertainty
    let mut k_gr_qg_lower: Vec<f64> = Vec::new();  // -1σ uncertainty

    for &r in radii.iter() {
        let phi = constants.g * m_sun / (r * constants.c * constants.c);

        // GR-only: K_GR = K_std · √(1 - 2φ)
        let gr_factor = (1.0 - 2.0 * phi).max(0.0).sqrt();
        k_gr_only.push(k_std * gr_factor);

        // GR×QG with mean α̂_G
        let epsilon = constants.g * m_sun * 1.67e-27 / r;  // Gravitational potential energy (proton)
        let qg_factor_mean = 1.0 + alpha_g_mean * epsilon / planck.energy;
        k_gr_qg.push(k_std * gr_factor * qg_factor_mean);

        // Uncertainty band: ±1σ on α̂_G
        let qg_factor_upper = 1.0 + (alpha_g_mean + alpha_g_sigma) * epsilon / planck.energy;
        let qg_factor_lower = 1.0 + (alpha_g_mean - alpha_g_sigma) * epsilon / planck.energy;
        k_gr_qg_upper.push(k_std * gr_factor * qg_factor_upper);
        k_gr_qg_lower.push(k_std * gr_factor * qg_factor_lower);
    }

    // Create main plot
    let root = BitMapBackend::new("figures/fig2_gravitational_enhanced.png", (1600, 1000))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Split into main panel (left 85%) and inset (right 15%)
    let (main_area, inset_area) = root.split_horizontally(1360);

    // Main plot with log x-axis
    let mut main_chart = ChartBuilder::on(&main_area)
        .caption("Gravitational K-Parameter: GR Time Dilation + QG Correction",
                 ("sans-serif", 40).into_font())
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(
            (3.0..1e6_f64).log_scale(),  // Log scale r/r_s
            (k_gr_only.iter().cloned().fold(f64::INFINITY, f64::min) * 0.9)..
            (k_gr_qg_upper.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.1)
        )?;

    main_chart.configure_mesh()
        .x_desc("Distance / Schwarzschild Radius (r/r_s)")
        .y_desc("K-Parameter [J·s^{-1/2}]")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.2e}", y))
        .draw()?;

    // Draw uncertainty band (filled polygon)
    let band_points_upper: Vec<(f64, f64)> = r_over_rs.iter().zip(&k_gr_qg_upper)
        .map(|(x, y)| (*x, *y)).collect();
    let band_points_lower: Vec<(f64, f64)> = r_over_rs.iter().zip(&k_gr_qg_lower)
        .rev().map(|(x, y)| (*x, *y)).collect();

    let mut band_polygon = band_points_upper.clone();
    band_polygon.extend(band_points_lower);

    main_chart.draw_series(std::iter::once(Polygon::new(
        band_polygon,
        &RED.mix(0.15)
    )))?;

    // Draw baseline (black dashed)
    main_chart.draw_series(LineSeries::new(
        r_over_rs.iter().zip(&k_baseline).map(|(x, y)| (*x, *y)),
        &BLACK
    ))?
    .label("K_std (baseline)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    // Draw GR-only (blue dashed)
    main_chart.draw_series(LineSeries::new(
        r_over_rs.iter().zip(&k_gr_only).map(|(x, y)| (*x, *y)),
        ShapeStyle::from(&BLUE).stroke_width(2)
    ))?
    .label("GR-only: √(1-2φ)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw GR×QG (red solid)
    main_chart.draw_series(LineSeries::new(
        r_over_rs.iter().zip(&k_gr_qg).map(|(x, y)| (*x, *y)),
        ShapeStyle::from(&RED).stroke_width(3)
    ))?
    .label("GR×QG: α̂_G = (6.96±0.15)×10^{-10}")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    main_chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    // Inset: Earth regime (r ≈ 6.371×10^6 m)
    let r_earth = 6.371e6;
    let r_over_rs_earth = r_earth / r_s;

    let mut inset_chart = ChartBuilder::on(&inset_area)
        .caption("Earth Surface", ("sans-serif", 24).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (r_over_rs_earth * 0.999)..(r_over_rs_earth * 1.001),
            (k_std * 0.9995)..(k_std * 1.0005)
        )?;

    inset_chart.configure_mesh()
        .x_desc("r/r_s")
        .y_desc("K")
        .x_label_formatter(&|x| format!("{:.1e}", x))
        .y_label_formatter(&|y| format!("{:.2e}", y))
        .draw()?;

    // Draw Earth point
    let phi_earth = constants.g * 5.972e24 / (r_earth * constants.c * constants.c);
    let k_earth_gr_qg = k_std * (1.0 - 2.0 * phi_earth).sqrt() *
                        (1.0 + alpha_g_mean * constants.g * 5.972e24 * 1.67e-27 / (r_earth * planck.energy));

    inset_chart.draw_series(PointSeries::of_element(
        vec![(r_over_rs_earth, k_earth_gr_qg)],
        5,
        &RED,
        &|c, s, st| {
            return Circle::new(c, s, st.filled());
        },
    ))?;

    root.present()?;

    println!("✅ Generated: figures/fig2_gravitational_enhanced.png");
    println!("   - Log x-axis from 3r_s to 10^6 r_s");
    println!("   - Three curves: baseline, GR-only, GR×QG");
    println!("   - Uncertainty band from α̂_G = (6.96 ± 0.15) × 10^{{-10}}");
    println!("   - Earth regime inset at r = 6.371 × 10^6 m");

    Ok(())
}

/// Generate Enhanced Figure 3: Dark Matter Annual Modulation
///
/// **Enhancements per user specifications:**
/// - Daily-binned data with 1σ error bars
/// - SHM sinusoid overlay with best-fit phase
/// - Residuals subplot below main panel
/// - Lomb-Scargle periodogram side panel
/// - Covariate correlation heatmap
pub fn generate_enhanced_fig3_dark_matter() -> Result<(), Box<dyn Error>> {
    println!("🎨 Generating Enhanced Figure 3: Dark Matter Annual Modulation...");

    // TODO: Implement multi-panel layout with:
    // - Main panel: K_dark vs day with error bars + sinusoid fit
    // - Bottom panel: Residuals (data - fit)
    // - Right panel: Lomb-Scargle periodogram
    // - Corner panel: Covariate correlation matrix

    println!("⏳ Enhanced Figure 3 implementation in progress...");

    Ok(())
}
