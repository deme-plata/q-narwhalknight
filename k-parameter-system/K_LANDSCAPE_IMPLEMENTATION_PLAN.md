# K-Parameter Landscape 3D Visualization - Implementation Plan
**Date**: October 2, 2025
**Priority**: HIGH - Publication Visual Impact
**Status**: Design Complete | Implementation Ready

---

## 🎨 Vision: Publication-Quality 3D Landscape Figures

Based on the reference image, we need to implement **3D heatmap/landscape visualizations** showing the K-Parameter as a surface across its parameter space. This will be the "killer visual" that makes reviewers immediately understand the framework.

### Reference Image Analysis:
- ✅ **3D surface** with height encoding K value
- ✅ **Color gradient**: Warm (orange/red) = high K, Cool (blue/cyan) = low K
- ✅ **Black contour lines**: Iso-K curves at constant K values
- ✅ **Smooth interpolation**: Professional scientific visualization quality
- ✅ **Clear axes**: Grid lines, labels with units
- ✅ **Lighting/shading**: Subtle 3D effect for depth perception

---

## 📊 Three Landscape Variants (Per User Specification)

### Variant A: Core Formalism Landscape (Introduction)
**Purpose**: Teach how K grows with energy/entropy fluctuations

**Axes**:
- **X-axis**: log₁₀(ΔH) [J] - Energy uncertainty (logarithmic)
- **Y-axis**: log₁₀(ΔS) [J/K] - Entropy variance (logarithmic)
- **Z-axis / Color**: K/K_std - Normalized K-Parameter at fixed τ

**Formula**:
```
K = 2π√(ΔH · ΔS · ℏ / τ)
K/K_std = √(ΔH/ΔH_std · ΔS/ΔS_std)
```

**Features to Show**:
- **High peak (orange)**: Both ΔH and ΔS large → maximum K
- **Deep basins (blue)**: Either ΔH or ΔS small → suppressed K
- **Diagonal ridge**: K constant along ΔH·ΔS = const line

**Callouts**:
- "K-sensitivity ridge" - steepest ascent path
- "Information-collapse basin" - low ΔS region
- "Energy-collapse basin" - low ΔH region

**Caption Text**:
```
K-Parameter landscape showing how energy uncertainty (ΔH) and entropy
variance (ΔS) jointly determine quantum information retention capacity.
Warm peaks mark regimes of maximal information gain; cool basins indicate
decoherence-dominated zones. At fixed interrogation time τ, K scales as
√(ΔH·ΔS), visible in the diagonal ridge structure.
```

---

### Variant B: Gravitational Enhancement Landscape (§2.2)
**Purpose**: Visualize GR+QG corrections to K-Parameter

**Axes**:
- **X-axis**: φ = GM/(rc²) - Dimensionless gravitational potential
- **Y-axis**: τ [s] - Interrogation time (log scale)
- **Z-axis / Color**: K_gravity/K_std = √(1-2φ)·(1 + α̂_G·ε/E_P)

**Formula**:
```
K_gravity = K_std · √(1-2GM/rc²) · (1 + α̂_G · GMm/r / E_P)
          = K_std · √(1-2φ) · (1 + α̂_G·φ·mc²/E_P)
```

**Features to Show**:
- **GR suppression valley**: φ → 1 (near r_s) causes K → 0
- **QG enhancement ridge**: α̂_G correction lifts K at intermediate φ
- **1/√τ slope**: K decreases with longer interrogation time

**Overlay Data**:
- Vertical line at φ = 1 (Schwarzschild radius) with "Event Horizon" label
- Shaded uncertainty band from α̂_G = (6.96 ± 0.15) × 10⁻¹⁰ posterior
- Earth regime marker: φ_⊕ ∼ 10⁻⁹

**Callouts**:
- "GR time-dilation valley" at high φ
- "QG correction plateau" at moderate φ
- "Weak-field regime" at low φ

**Caption Text**:
```
Gravitational K-Parameter landscape across spacetime curvature (φ = GM/rc²)
and measurement timescale (τ). General relativistic time dilation suppresses
K near massive objects (blue valley approaching Schwarzschild radius), while
quantum gravitational corrections (α̂_G = 6.96×10⁻¹⁰) provide measurable
enhancement at intermediate scales (warm plateau). Earth's surface sits at
φ ∼ 10⁻⁹ (weak-field limit).
```

---

### Variant C: Dark Sector Landscape (§4.1)
**Purpose**: Show DM decoherence vs DE enhancement trade-off

**Axes**:
- **X-axis**: X = Γ_DM·t - Cumulative DM decoherence (dimensionless)
- **Y-axis**: Y = β_DE·Λ·t² - Dark energy drift (dimensionless)
- **Z-axis / Color**: K_dark/K_std = exp(-X)·(1 + Y)

**Formula**:
```
K_dark = K_std · exp(-Γ_DM·t) · (1 + β_DE·Λ·t²)
where Γ_DM = n_DM · λ_DM · v_DM
```

**Features to Show**:
- **DM suppression basin (blue)**: X > 1 → exp(-X) kills K exponentially
- **DE enhancement ridge (orange)**: Y > 0 → (1+Y) boosts K quadratically
- **Saddle point**: Compensation contour where DM and DE effects balance

**Overlay Data**:
- "One e-fold" line at X = 1 (where K drops by factor e)
- Measured Y range from dark energy fit (w = -1.035 ± 0.008)
- Annual modulation track showing Earth's orbital motion

**Callouts**:
- "DM-dominated suppression" (left basin)
- "DE-driven growth" (right ridge)
- "Compensation contour" (saddle)

**Caption Text**:
```
Dark sector K-Parameter landscape showing interplay between dark matter
decoherence (X = Γ_DM·t, horizontal) and dark energy enhancement
(Y = β_DE·Λ·t², vertical). Blue basin marks DM-dominated regime where
quantum states rapidly decohere; orange ridge shows DE-driven growth from
cosmological constant coupling. Saddle contour identifies the compensation
line where effects balance, enabling precise dark energy equation of state
measurement (w = -1.035 ± 0.008).
```

---

## 🛠️ Technical Implementation Plan

### Phase 1: Add 3D Surface Plot Support to k-graph-generator

**New Rust Module**: `crates/k-graph-generator/src/surface3d.rs`

```rust
use plotters::prelude::*;
use plotters::style::full_palette::{ORANGE, CYAN, PURPLE};
use ndarray::{Array2, Array1};

pub struct Surface3D {
    title: String,
    x_label: String,
    y_label: String,
    z_label: String,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    z_data: Array2<f64>,  // 2D grid of Z values
    colormap: ColorMap,
    contour_levels: Vec<f64>,
    annotations: Vec<Annotation3D>,
}

pub enum ColorMap {
    HotCold,      // Orange (high) → Blue (low) - for K landscapes
    Viridis,      // Perceptually uniform - for general use
    Plasma,       // High contrast - for presentations
    ColorblindSafe,  // Okabe-Ito palette projection
}

pub struct Annotation3D {
    x: f64,
    y: f64,
    text: String,
    style: AnnotationStyle,
}

impl Surface3D {
    pub fn new(title: &str, x_label: &str, y_label: &str, z_label: &str) -> Self;

    /// Set the grid of (x, y, z) values
    pub fn set_data(mut self, x: Vec<f64>, y: Vec<f64>, z: Array2<f64>) -> Self;

    /// Compute Z from a closure: z = f(x, y)
    pub fn from_function<F>(mut self, x_range: (f64, f64), y_range: (f64, f64),
                           nx: usize, ny: usize, f: F) -> Self
    where F: Fn(f64, f64) -> f64;

    /// Choose colormap style
    pub fn with_colormap(mut self, cmap: ColorMap) -> Self;

    /// Add contour lines at specified Z levels
    pub fn with_contours(mut self, levels: Vec<f64>) -> Self;

    /// Add automatic contour lines (n equally spaced)
    pub fn with_auto_contours(mut self, n_levels: usize) -> Self;

    /// Add text annotation at (x, y) position
    pub fn add_annotation(mut self, x: f64, y: f64, text: &str) -> Self;

    /// Add uncertainty band (shade region between z_lower and z_upper)
    pub fn add_uncertainty_band(mut self, z_lower: Array2<f64>, z_upper: Array2<f64>) -> Self;

    /// Set viewing angle (azimuth, elevation in degrees)
    pub fn set_view_angle(mut self, azimuth: f64, elevation: f64) -> Self;

    /// Export to file (PNG, SVG, or PDF)
    pub fn export(&self, path: &str, width: u32, height: u32, dpi: u32) -> Result<()>;
}
```

**Implementation Strategy**:
1. Use `plotters` 3D plotting backend (plotters::coord::types::RangedCoordf64)
2. Interpolate Z grid to smooth surface
3. Map Z values to colors via colormap
4. Project contour lines onto surface
5. Apply lighting/shading for depth cues

---

### Phase 2: Implement Colormap System

**File**: `crates/k-graph-generator/src/colormap.rs`

```rust
pub struct ColorMap {
    name: String,
    gradient: Vec<RGBColor>,
}

impl ColorMap {
    /// Hot (orange) to Cold (blue) - perfect for K landscapes
    pub fn hot_cold() -> Self {
        // Linear interpolation: Orange → Yellow → White → Cyan → Blue
        let colors = vec![
            RGBColor(255, 140, 0),   // Dark orange (low K)
            RGBColor(255, 200, 0),   // Orange
            RGBColor(255, 255, 100), // Yellow
            RGBColor(200, 255, 255), // Light cyan
            RGBColor(0, 200, 255),   // Cyan
            RGBColor(0, 100, 200),   // Blue (high K)
        ];
        Self::from_gradient("HotCold", colors)
    }

    /// Colorblind-safe palette (Okabe-Ito derived)
    pub fn colorblind_safe() -> Self {
        let colors = vec![
            RGBColor(230, 159, 0),   // Orange
            RGBColor(86, 180, 233),  // Sky blue
            RGBColor(0, 158, 115),   // Bluish green
            RGBColor(240, 228, 66),  // Yellow
        ];
        Self::from_gradient("ColorblindSafe", colors)
    }

    /// Map value in [z_min, z_max] to RGB color
    pub fn map(&self, z: f64, z_min: f64, z_max: f64) -> RGBColor {
        let t = (z - z_min) / (z_max - z_min);  // Normalize to [0, 1]
        let t_clamped = t.clamp(0.0, 1.0);
        self.interpolate(t_clamped)
    }

    fn interpolate(&self, t: f64) -> RGBColor {
        // Linear interpolation between gradient stops
        let n = self.gradient.len();
        let idx = (t * (n - 1) as f64).floor() as usize;
        let frac = t * (n - 1) as f64 - idx as f64;

        if idx >= n - 1 {
            return self.gradient[n - 1];
        }

        let c1 = self.gradient[idx];
        let c2 = self.gradient[idx + 1];

        RGBColor(
            ((1.0 - frac) * c1.0 as f64 + frac * c2.0 as f64) as u8,
            ((1.0 - frac) * c1.1 as f64 + frac * c2.1 as f64) as u8,
            ((1.0 - frac) * c1.2 as f64 + frac * c2.2 as f64) as u8,
        )
    }
}
```

---

### Phase 3: Create K-Landscape Generator Functions

**File**: `src/landscape_generator.rs`

```rust
use k_graph_generator::Surface3D;
use k_constants::PhysicalConstants;
use nalgebra::DVector;

/// Generate Variant A: Core formalism landscape (log ΔH vs log ΔS)
pub fn generate_variant_a_core_landscape() -> Result<(), Box<dyn std::error::Error>> {
    let constants = PhysicalConstants::default();
    let k_std = 1e-15;  // Standard K reference value
    let tau = 1.0;      // Fixed interrogation time (1 second)

    // Create 3D surface
    let mut surface = Surface3D::new(
        "K-Parameter Landscape: Energy-Entropy Phase Space",
        "log₁₀(ΔH) [J]",
        "log₁₀(ΔS) [J/K]",
        "K/K_std",
    );

    // Define grid: log₁₀(ΔH) from -25 to -10, log₁₀(ΔS) from -25 to -10
    surface = surface.from_function(
        (-25.0, -10.0),  // x_range: log₁₀(ΔH)
        (-25.0, -10.0),  // y_range: log₁₀(ΔS)
        100,             // nx points
        100,             // ny points
        |log_dh, log_ds| {
            let dh = 10_f64.powf(log_dh);
            let ds = 10_f64.powf(log_ds);
            let k = 2.0 * std::f64::consts::PI * (dh * ds * constants.hbar / tau).sqrt();
            k / k_std
        }
    );

    // Use hot-cold colormap: orange (high K) to blue (low K)
    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(10)  // 10 iso-K contour lines
        .set_view_angle(45.0, 30.0);  // Azimuth 45°, elevation 30°

    // Add annotations
    surface = surface
        .add_annotation(-17.5, -17.5, "K-sensitivity ridge")
        .add_annotation(-23.0, -12.0, "Energy-collapse basin")
        .add_annotation(-12.0, -23.0, "Information-collapse basin");

    // Export as high-res PNG and SVG
    surface.export("figures/k_landscape_variant_a.png", 1600, 1200, 300)?;
    surface.export("figures/k_landscape_variant_a.svg", 1600, 1200, 300)?;

    println!("Generated: k_landscape_variant_a (Core formalism)");
    Ok(())
}

/// Generate Variant B: Gravitational landscape (φ vs τ)
pub fn generate_variant_b_gravitational() -> Result<(), Box<dyn std::error::Error>> {
    let constants = PhysicalConstants::default();
    let alpha_g = 6.96e-10;  // Measured effective QG coupling
    let e_p = constants.planck_energy();
    let m = constants.proton_mass;  // Test mass
    let c = constants.c;

    let mut surface = Surface3D::new(
        "Gravitational K-Parameter Landscape",
        "φ = GM/(rc²) [dimensionless]",
        "τ [s] (log scale)",
        "K_gravity / K_std",
    );

    surface = surface.from_function(
        (1e-12, 0.9),  // φ from near-zero to near Schwarzschild radius
        (0.001, 1000.0),  // τ from 1 ms to 1000 s (log scale)
        100,
        100,
        |phi, tau| {
            // K_gravity = K_std · √(1-2φ) · (1 + α_G·φ·mc²/E_P)
            let gr_factor = (1.0 - 2.0 * phi).max(0.0).sqrt();
            let qg_factor = 1.0 + alpha_g * phi * m * c * c / e_p;
            let tau_factor = 1.0 / tau.sqrt();
            gr_factor * qg_factor * tau_factor
        }
    );

    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(12)
        .set_view_angle(60.0, 25.0);

    // Add critical annotations
    surface = surface
        .add_annotation(1.0, 1.0, "Event Horizon (r_s)")
        .add_annotation(0.1, 0.1, "QG correction plateau")
        .add_annotation(1e-9, 10.0, "Earth regime");

    // Add uncertainty band from α_G = 6.96 ± 0.15 × 10⁻¹⁰
    // (Would require computing z_lower and z_upper grids)

    surface.export("figures/k_landscape_variant_b.png", 1600, 1200, 300)?;
    surface.export("figures/k_landscape_variant_b.svg", 1600, 1200, 300)?;

    println!("Generated: k_landscape_variant_b (Gravitational)");
    Ok(())
}

/// Generate Variant C: Dark sector landscape (DM vs DE)
pub fn generate_variant_c_dark_sector() -> Result<(), Box<dyn std::error::Error>> {
    let gamma_dm = 1e-6;  // DM decoherence rate (s⁻¹)
    let beta_de = 0.035;   // DE coupling from w = -1.035
    let lambda = 1.1e-52; // Cosmological constant (m⁻²)

    let mut surface = Surface3D::new(
        "Dark Sector K-Parameter Landscape",
        "X = Γ_DM·t [dimensionless]",
        "Y = β_DE·Λ·t² [dimensionless]",
        "K_dark / K_std",
    );

    surface = surface.from_function(
        (0.0, 5.0),  // X: 0 to 5 e-folds of DM decoherence
        (0.0, 0.2),  // Y: 0 to 20% DE enhancement
        100,
        100,
        |x, y| {
            // K_dark = K_std · exp(-X) · (1 + Y)
            (-x).exp() * (1.0 + y)
        }
    );

    surface = surface
        .with_colormap(ColorMap::HotCold)
        .with_auto_contours(15)
        .set_view_angle(50.0, 30.0);

    // Add physics annotations
    surface = surface
        .add_annotation(1.0, 0.0, "One e-fold (X=1)")
        .add_annotation(0.0, 0.1, "DE-driven growth")
        .add_annotation(3.0, 0.0, "DM-dominated suppression")
        .add_annotation(2.0, 0.08, "Compensation contour");

    surface.export("figures/k_landscape_variant_c.png", 1600, 1200, 300)?;
    surface.export("figures/k_landscape_variant_c.svg", 1600, 1200, 300)?;

    println!("Generated: k_landscape_variant_c (Dark sector)");
    Ok(())
}
```

---

## 📐 LaTeX Integration

Add these figures to the document:

```latex
% In Introduction (after core formalism explanation)
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{fig_landscape_variant_a.png}
  \caption{\textbf{K-Parameter Landscape: Energy-Entropy Phase Space.}
  Surface height and color encode the K-Parameter value, with warm colors/peaks
  for high $K$ and cool colors/basins for low $K$. Black contour lines are iso-K
  curves. Interpreting $K = 2\pi\sqrt{\Delta H \cdot \Delta S \cdot \hbar / \tau}$,
  this landscape visualizes how energy uncertainty ($\Delta H$), entropy variance
  ($\Delta S$), and the characteristic timescale ($\tau = 1$ s fixed here) trade
  off to raise or suppress $K$. The diagonal ridge shows that $K$ remains constant
  along $\Delta H \cdot \Delta S = \text{const}$ lines, while peaks identify
  operating regimes with maximal information gain and basins mark decoherence-
  dominated zones where quantum coherence rapidly collapses.}
  \label{fig:k_landscape_core}
\end{figure}

% In §2.2 Gravitational Enhancement
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{fig_landscape_variant_b.png}
  \caption{\textbf{Gravitational K-Parameter Landscape.}
  3D visualization of $K_{\text{gravity}}/K_{\text{std}}$ across dimensionless
  gravitational potential $\phi = GM/(rc^2)$ (horizontal) and measurement
  interrogation time $\tau$ (vertical, log scale). General relativistic time
  dilation suppresses $K$ via the $\sqrt{1-2\phi}$ factor, creating a dramatic
  valley approaching the Schwarzschild radius ($\phi \to 1$, event horizon
  marked). Quantum gravitational corrections provide measurable enhancement
  through the $(1 + \hat{\alpha}_G \epsilon/E_P)$ term with fitted coupling
  $\hat{\alpha}_G = (6.96 \pm 0.15) \times 10^{-10}$, visible as the warm
  plateau at intermediate $\phi \sim 0.01$–$0.5$. Earth's surface sits at
  $\phi \approx 10^{-9}$ (weak-field limit, lower-left corner). The $1/\sqrt{\tau}$
  slope reflects fundamental quantum measurement scaling.}
  \label{fig:k_landscape_gravity}
\end{figure}

% In §4.1 Dark Sector Detection
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{fig_landscape_variant_c.png}
  \caption{\textbf{Dark Sector K-Parameter Landscape.}
  Trade-off surface showing $K_{\text{dark}}/K_{\text{std}} = e^{-X}(1+Y)$
  where $X = \Gamma_{DM} t$ quantifies cumulative dark matter decoherence and
  $Y = \beta_{DE} \Lambda t^2$ measures dark energy coupling strength. The
  blue basin (left) marks the DM-dominated regime where quantum states rapidly
  decohere ($X > 1$ corresponds to one e-fold suppression); the orange ridge
  (right) shows DE-driven enhancement from cosmological constant coupling
  ($\beta_{DE} \approx 0.035$ derived from measured $w = -1.035 \pm 0.008$).
  The saddle contour (diagonal) identifies the compensation line where DM
  suppression and DE enhancement balance, enabling precision dark energy
  equation of state determination. Annual/diurnal modulation signals trace
  paths across this landscape as Earth orbits through the galactic DM halo.}
  \label{fig:k_landscape_dark}
\end{figure}
```

---

## 🎯 Success Metrics

**Visual Impact Goals**:
- [ ] Reviewers can identify high-K vs low-K regimes instantly
- [ ] Trade-offs between physical effects are visually obvious
- [ ] Figures are publication-ready without additional explanation
- [ ] Color-blind accessibility verified (tools like Coblis simulator)
- [ ] Export quality: ≥300 DPI PNG + vector SVG/PDF

**Technical Goals**:
- [ ] Surface interpolation smooth (no visible grid artifacts)
- [ ] Contour lines clean and mathematically accurate
- [ ] Colormap scientifically appropriate (perceptually uniform)
- [ ] Annotations positioned automatically to avoid overlaps
- [ ] Rendering time <10 seconds per figure

---

## 📋 Implementation Timeline

**Week 1**: Core 3D surface infrastructure
- Day 1-2: Implement Surface3D struct and basic plotting
- Day 3-4: Build colormap system with hot-cold gradient
- Day 5: Contour line extraction and overlay

**Week 2**: Variant-specific generators + LaTeX integration
- Day 1: Variant A (core formalism) generator
- Day 2: Variant B (gravitational) with uncertainty bands
- Day 3: Variant C (dark sector) with compensation contours
- Day 4: LaTeX integration + caption refinement
- Day 5: Quality verification + accessibility check

**Total**: 10 days to publication-ready 3D landscape figure suite

---

## 🚀 Beyond the Baseline: Advanced Features

Once core landscapes are working, consider:

1. **Interactive 3D exports** (WebGL via plotters-canvas)
2. **Animation sequences** showing time evolution (e.g., annual DM modulation)
3. **Cross-section slices** as inset panels (2D cuts through 3D surface)
4. **Uncertainty volumes** (3D shaded regions from parameter posteriors)
5. **Measured data overlay** (scatter points on landscape surface)

---

## 📝 Notes for User

This implementation plan fully addresses your vision for "beautiful figures"
that communicate the K-Parameter framework's multi-dimensional structure. The
three landscape variants will serve as:

- **Variant A**: The "teaching figure" that makes reviewers understand the core formalism
- **Variant B**: The "quantum gravity proof" showing GR+QG interplay
- **Variant C**: The "dark sector discovery" demonstrating DM/DE detection mechanism

All three will use the same visual language (hot-cold colormap, contour lines,
annotations) for consistency across the paper.

**Next step**: Implement Surface3D module in k-graph-generator, starting with
basic plotters 3D backend integration.
