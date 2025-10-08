# K-Parameter Landscape Visualization - Complete Specification
**Date**: October 2, 2025
**Priority**: HIGH - Core pedagogical figure for paper
**Status**: Design Complete - Ready for Implementation

---

## 🎯 Vision: 3D K-Parameter Landscape Heatmaps

Generate publication-quality 3D surface/heatmap visualizations showing the K-Parameter as a **landscape** with:
- **Color gradient**: Warm (orange/red) = high K → Cool (blue/purple) = low K
- **Surface height**: Encodes K value in 3D perspective
- **Contour lines**: Black iso-K curves showing constant K
- **Annotations**: Callouts for key features (peaks, ridges, basins)
- **Multiple variants**: Different parameterizations for each physics section

**Pedagogical Goal**: Make reviewers immediately grasp the multi-dimensional K-Parameter space and understand how different physical effects create peaks, ridges, and basins in the landscape.

---

## 📊 Three Landscape Variants (One per Section)

### **Variant A: Core Formalism Landscape (for §1-2 Introduction)**

**Purpose**: Teach how K grows with energy uncertainty AND entropy variance

**Axes**:
- X-axis: log₁₀(ΔH) [J] - Energy uncertainty
- Y-axis: log₁₀(ΔS) [J/K] - Entropy variance
- Z-axis (color/height): K/K_std - Normalized K-Parameter at fixed τ

**Mathematical Model**:
```
K(ΔH, ΔS) = (2π/τ) × √(ΔH · ΔS · ℏ)
K_normalized = K / K_std
```

**Key Features to Highlight**:
1. **High peak (NE corner)**: Both ΔH and ΔS large → maximum K
   - Label: "Maximum Information Regime"
   - Color: Deep orange/red

2. **Two ridges**:
   - East ridge: High ΔS, moderate ΔH
   - North ridge: High ΔH, moderate ΔS
   - Label: "Energy-Dominated" and "Entropy-Dominated" paths

3. **Deep basin (SW corner)**: Low ΔH AND low ΔS → minimum K
   - Label: "Noise Floor"
   - Color: Deep blue

4. **Diagonal saddle**: ΔH·ΔS ≈ constant (iso-K contour)
   - Label: "Energy-Entropy Trade-off"

**Range**:
- ΔH: 10⁻²⁵ to 10⁻¹⁵ J (10 orders of magnitude)
- ΔS: 10⁻²⁵ to 10⁻²⁰ J/K (5 orders of magnitude)
- K/K_std: 0.01 to 100 (4 orders of magnitude, use log colorscale)

**Contour Levels**: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] × K_std

**Caption**:
> **K-Parameter Landscape: Core Formalism.** Surface height and color encode K/K_std with warm peaks (high K) and cool basins (low K). Black contours are iso-K curves. The landscape shows how K = (2π/τ)√(ΔH·ΔS·ℏ) grows along both energy-uncertainty (x-axis) and entropy-variance (y-axis) directions. The NE peak marks the maximum-information regime where sensors have maximal sensitivity. The SW basin is the noise floor where measurements lack statistical power. Diagonal contours reveal the fundamental energy-entropy trade-off in quantum parameter estimation.

---

### **Variant B: Gravitational Landscape (for §2.2 Quantum Gravity)**

**Purpose**: Visualize GR time dilation + quantum gravity corrections vs potential and measurement time

**Axes**:
- X-axis: φ = GM/(rc²) - Dimensionless gravitational potential (0 to 0.5, with Schwarzschild φ_s = 0.5)
- Y-axis: τ [s] - Interrogation time (10⁻⁶ to 10³ s, log scale)
- Z-axis (color/height): K_gravity/K_std

**Mathematical Model**:
```
K_gravity = K_std × √(1 - 2φ) × (1 + α̂_G · ε/E_P)

where:
  φ = GM/(rc²)  (dimensionless potential)
  ε = GMm/r     (potential energy)
  α̂_G = (6.96 ± 0.15) × 10⁻¹⁰  (fitted QG coupling)
  E_P = 1.22 × 10¹⁹ GeV        (Planck energy)

K_gravity/K_std = √(1 - 2φ) × (1 + α̂_G · φ · mc²/E_P)
```

**Key Features to Highlight**:
1. **GR suppression ridge** (φ-axis):
   - Gentle downward slope as φ → φ_s
   - Label: "GR Time Dilation √(1-2φ)"
   - Vertical line at φ = 0.5 (Schwarzschild radius)

2. **QG enhancement bump** (small φ):
   - Slight uplift from baseline at φ ≈ 0.01-0.1
   - Label: "QG Correction (α̂_G · ε/E_P)"
   - Annotate with α̂_G value

3. **Time-scaling slope** (τ-axis):
   - K ∝ 1/√τ creates gentle downward slope
   - Label: "Interrogation Time Scaling"

4. **Optimal operating point**:
   - Mark (φ ≈ 10⁻⁸, τ ≈ 10⁻² s) - Earth surface, BEC timescales
   - Label: "BEC Gradiometer Regime"

**Range**:
- φ: 0 to 0.45 (stop before singularity at 0.5)
- τ: 10⁻⁶ to 10³ s (9 orders of magnitude)
- K/K_std: 0.8 to 1.2 (GR suppression vs QG enhancement)

**Contour Levels**: [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15] × K_std

**Annotations**:
- Vertical line at φ = 0.5: "r = r_s (event horizon)"
- Shaded region φ > 0.45: "Strong-field regime (inaccessible)"
- Callout at φ ≈ 10⁻⁹: "Earth surface (2GM_⊕/r_⊕c²)"

**Caption**:
> **K-Parameter Gravitational Landscape.** Color/height shows K_gravity/K_std as a function of dimensionless gravitational potential φ = GM/(rc²) and interrogation time τ. The landscape combines general relativistic time dilation (√(1-2φ) suppression, dominant at large φ) with quantum gravitational corrections (1 + α̂_G·ε/E_P enhancement, fitted α̂_G = 6.96×10⁻¹⁰). The gentle φ-ridge shows GR effects; the slight uplift at small φ reveals QG modifications. The Schwarzschild radius (φ = 0.5, vertical line) marks the classical event horizon. Earth-surface conditions (φ ≈ 10⁻⁹, marked) lie in the nearly-flat Newtonian regime where QG corrections are detectable but small.

---

### **Variant C: Dark Sector Landscape (for §4.1 Dark Matter & Dark Energy)**

**Purpose**: Show competition between DM decoherence (suppression) and DE growth (enhancement)

**Axes**:
- X-axis: X = Γ_DM·t - Cumulative dark matter decoherence (0 to 5 e-folds)
- Y-axis: Y = β_DE·Λ·t² - Dark energy drift (0 to 0.2)
- Z-axis (color/height): K_dark/K_std

**Mathematical Model**:
```
K_dark = K_std × exp(-X) × (1 + Y)

where:
  X = Γ_DM·t  (DM decoherence: Γ_DM = n_DM · λ_DM · v_DM)
  Y = β_DE·Λ·t²  (DE coupling)

Γ_DM = (0.3 GeV/cm³)/(m_DM) × λ_DM × (220 km/s)
β_DE ≈ 10⁻⁶ (fitted)
Λ = 1.11 × 10⁻⁵² m⁻²
```

**Key Features to Highlight**:
1. **DM suppression basin** (large X, Y ≈ 0):
   - Exponential decay along X-axis
   - Deep blue valley at X > 2
   - Label: "DM-Dominated Decoherence"
   - Tick marks at X = 1 ("one e-fold")

2. **DE enhancement ridge** (X ≈ 0, large Y):
   - Gentle uplift along Y-axis
   - Orange plateau at Y > 0.1
   - Label: "DE-Driven Growth"

3. **Compensation contour** (diagonal):
   - Line where exp(-X)·(1+Y) ≈ 1
   - Label: "DM-DE Balance (K = K_std)"
   - This is a saddle point

4. **Optimal measurement window**:
   - Region X < 1, Y < 0.05
   - Label: "Observable Modulation Regime"
   - Mark experimental data points

**Range**:
- X: 0 to 5.0 (5 e-folding times)
- Y: 0 to 0.2 (20% DE enhancement)
- K/K_std: 0.01 to 1.2 (DM can suppress by 100×, DE enhances by 20%)

**Contour Levels**: [0.05, 0.1, 0.2, 0.37 (1/e), 0.5, 0.8, 1.0, 1.1, 1.15] × K_std

**Annotations**:
- Vertical line at X = 1: "Γ_DM·t = 1"
- Horizontal line at Y = 0.07: "Measured β_DE·Λ·t²"
- Shaded region X > 3: "Signal Lost to DM"
- Arrow showing "Annual modulation path" (ellipse in X-Y plane)

**Caption**:
> **K-Parameter Dark Sector Landscape.** Color/height encodes K_dark/K_std showing the competition between dark matter decoherence (exp(-X) suppression along x-axis, X = Γ_DM·t) and dark energy coupling (1+Y enhancement along y-axis, Y = β_DE·Λ·t²). The left basin (blue, large X) is DM-dominated suppression; the right ridge (orange, large Y) is DE-driven growth. The diagonal compensation contour (K = K_std, dashed) marks the DM-DE balance. Our 15-site underground sensor network operates in the X < 1, Y < 0.05 regime (shaded), where annual/diurnal modulations from Earth's motion through the DM halo are detectable with 5.1σ significance.

---

## 🎨 Visualization Design Specifications

### Color Palette (Color-Blind Safe)
**Use viridis or plasma colormap** (matplotlib standard, perceptually uniform):
- **High K**: Yellow/Orange (#FDE724, #E16462)
- **Mid K**: Green/Cyan (#35B779, #31688E)
- **Low K**: Blue/Purple (#440154, #21918C)

**Alternative**: Use ColorBrewer "RdYlBu" reversed (red-yellow-blue):
- High K: Red (#D73027)
- Mid K: Yellow (#FEE090)
- Low K: Blue (#4575B4)

**Colorbar**:
- Position: Right side, vertical
- Label: "K/K_std" (with units if using raw K)
- Tick marks at key values (0.1, 1.0, 10.0 for log scale)
- Continuous gradient, not discrete bins

### Contour Lines
- **Color**: Black (#000000)
- **Line width**: 0.5 pt (thin)
- **Line style**: Solid for major contours, dashed for half-intervals
- **Labels**: Inline contour labels at 2-3 positions per line
- **Levels**: Logarithmically spaced for wide dynamic range

### 3D Perspective vs 2D Heatmap
**Recommendation**: Provide **both** for each variant:

**2D Heatmap (Top View)**:
- Simpler to read quantitatively
- Easier to place annotations
- Better for print (vector export)
- Use this as primary figure

**3D Surface (Perspective View)**:
- More visually striking
- Better for talks/presentations
- Shows "landscape" metaphor clearly
- Use as supplementary or for specific highlights

**Implementation**: Generate both, let user choose or provide both as Fig Xa (2D) and Fig Xb (3D)

### Annotations & Callouts
**Arrow style**:
- Solid black arrows with small heads
- Text in sans-serif font, 10pt
- Background: white box with 80% opacity for readability
- Examples: "K-sensitivity ridge →", "← Decoherence basin"

**Markers for special points**:
- White circles with black outline for peaks/valleys
- Star marker for optimal operating points
- Cross (×) for experimental data points

### Grid & Axes
**Axes**:
- X-axis label: Full variable name + units, e.g., "log₁₀(ΔH) [J]"
- Y-axis label: Same convention
- Tick marks: 5-7 major ticks per axis
- Minor ticks: 4 between each major tick for log scales
- Grid: Major grid only, light gray (#CCCCCC), behind data

**Aspect ratio**: 1.2:1 (slightly wider than tall) for better landscape view

### Export Settings
- **Resolution**: 600 DPI minimum for raster (PNG)
- **Format**:
  - Primary: PDF (vector) for publication
  - Secondary: PNG (high-res) for presentations
  - Optional: SVG (for web)
- **Size**: 1200×1000 pixels minimum
- **Font embedding**: Ensure all fonts embedded in PDF

---

## 🛠️ Implementation Architecture

### New Rust Module: `k-landscape-generator`

**Location**: `crates/k-landscape-generator/`

**Dependencies**:
```toml
[dependencies]
plotters = { workspace = true }
plotters-backend = "0.3"
ndarray = { workspace = true }
nalgebra = { workspace = true }
k-constants = { path = "../k-constants" }
k-quantum-gravity = { path = "../k-quantum-gravity" }
k-dark-sector = { path = "../k-dark-sector" }
colorgrad = "0.6"  # For perceptually uniform colormaps
```

### Core Data Structure

```rust
/// Configuration for K-Parameter landscape visualization
pub struct LandscapeConfig {
    /// Variant type (Core, Gravity, DarkSector)
    pub variant: LandscapeVariant,

    /// X-axis configuration
    pub x_axis: AxisConfig,

    /// Y-axis configuration
    pub y_axis: AxisConfig,

    /// Colormap settings
    pub colormap: ColormapConfig,

    /// Contour settings
    pub contours: ContourConfig,

    /// Annotations to add
    pub annotations: Vec<Annotation>,

    /// Output settings
    pub output: OutputConfig,
}

pub enum LandscapeVariant {
    Core {
        tau: f64,  // Fixed interrogation time
    },
    Gravity {
        mass: f64,  // Central mass (kg)
    },
    DarkSector {
        dm_params: DarkMatterParams,
        de_params: DarkEnergyParams,
    },
}

pub struct AxisConfig {
    pub variable: Variable,
    pub range: (f64, f64),
    pub scale: Scale,  // Linear or Log
    pub label: String,
    pub units: String,
}

pub enum Variable {
    LogDeltaH,
    LogDeltaS,
    GravitationalPotential,  // φ
    InterrogationTime,       // τ
    DMDecoherence,           // X
    DEDrift,                 // Y
}

pub struct ColormapConfig {
    pub name: Colormap,  // Viridis, Plasma, RdYlBu
    pub scale: ColorScale,  // Linear or Log
    pub range: (f64, f64),  // K/K_std min and max
}

pub struct ContourConfig {
    pub levels: Vec<f64>,
    pub color: RGB,
    pub line_width: f64,
    pub show_labels: bool,
}

pub struct Annotation {
    pub position: (f64, f64),  // (x, y) in data coords
    pub text: String,
    pub arrow_to: Option<(f64, f64)>,  // If Some, draw arrow
    pub style: AnnotationStyle,
}

pub struct OutputConfig {
    pub path: String,
    pub format: OutputFormat,  // PDF, PNG, SVG, Both
    pub dpi: u32,
    pub width: u32,
    pub height: u32,
}
```

### Core Algorithm

```rust
impl LandscapeGenerator {
    /// Generate K-Parameter landscape
    pub fn generate(&self, config: &LandscapeConfig) -> Result<(), Box<dyn Error>> {
        // 1. Create mesh grid
        let (x_grid, y_grid) = self.create_mesh_grid(
            &config.x_axis,
            &config.y_axis,
            100  // 100×100 resolution
        );

        // 2. Compute K/K_std at each grid point
        let k_values = self.compute_k_landscape(
            &config.variant,
            &x_grid,
            &y_grid
        );

        // 3. Normalize to K/K_std
        let k_normalized = k_values / self.compute_k_std(&config.variant);

        // 4. Create 2D heatmap
        self.plot_heatmap(
            &x_grid,
            &y_grid,
            &k_normalized,
            config
        )?;

        // 5. Add contour lines
        self.add_contours(
            &x_grid,
            &y_grid,
            &k_normalized,
            &config.contours
        )?;

        // 6. Add annotations
        for annotation in &config.annotations {
            self.add_annotation(annotation)?;
        }

        // 7. Add colorbar
        self.add_colorbar(&config.colormap)?;

        // 8. Export
        self.export(&config.output)?;

        Ok(())
    }

    /// Compute K-Parameter landscape based on variant
    fn compute_k_landscape(
        &self,
        variant: &LandscapeVariant,
        x_grid: &Array2<f64>,
        y_grid: &Array2<f64>,
    ) -> Array2<f64> {
        match variant {
            LandscapeVariant::Core { tau } => {
                // x = log10(ΔH), y = log10(ΔS)
                let delta_h = x_grid.mapv(|x| 10_f64.powf(x));
                let delta_s = y_grid.mapv(|y| 10_f64.powf(y));

                // K = (2π/τ) × √(ΔH · ΔS · ℏ)
                let hbar = self.constants.hbar;
                ((2.0 * PI / tau) * (delta_h * delta_s * hbar).mapv(f64::sqrt))
            }

            LandscapeVariant::Gravity { mass } => {
                // x = φ = GM/(rc²), y = τ
                let phi = x_grid.clone();
                let tau = y_grid.clone();

                // K_gravity = K_std × √(1 - 2φ) × (1 + α̂_G · φ · mc²/E_P)
                let alpha_g = 6.96e-10;
                let e_p = 1.22e19 * 1.6e-10;  // Planck energy in J
                let m = 1.67e-27;  // Proton mass (test particle)
                let c = self.constants.c;

                let gr_factor = (1.0 - 2.0 * &phi).mapv(f64::sqrt);
                let qg_factor = 1.0 + alpha_g * &phi * m * c.powi(2) / e_p;

                let k_std = 1.0;  // Placeholder - compute based on params
                k_std * gr_factor * qg_factor / tau.mapv(f64::sqrt)
            }

            LandscapeVariant::DarkSector { dm_params, de_params } => {
                // x = X = Γ_DM·t, y = Y = β_DE·Λ·t²
                let x = x_grid.clone();
                let y = y_grid.clone();

                // K_dark = K_std × exp(-X) × (1 + Y)
                let k_std = 1.0;
                k_std * x.mapv(|xi| (-xi).exp()) * (1.0 + &y)
            }
        }
    }
}
```

### Contour Line Generation

Use **marching squares algorithm** (available in plotters or implement):

```rust
/// Generate contour lines using marching squares
fn generate_contours(
    &self,
    x_grid: &Array2<f64>,
    y_grid: &Array2<f64>,
    z_values: &Array2<f64>,
    levels: &[f64],
) -> Vec<Contour> {
    let mut contours = Vec::new();

    for &level in levels {
        let contour = marching_squares::contour(
            x_grid,
            y_grid,
            z_values,
            level
        );
        contours.push(contour);
    }

    contours
}
```

---

## 📋 Integration into visualization_generator.rs

Add three new functions:

```rust
/// Generate Figure 1b: K-Parameter Core Formalism Landscape
pub fn generate_fig1b_core_landscape() -> Result<(), Box<dyn Error>> {
    let config = LandscapeConfig {
        variant: LandscapeVariant::Core { tau: 1e-3 },  // 1 ms
        x_axis: AxisConfig {
            variable: Variable::LogDeltaH,
            range: (-25.0, -15.0),  // 10⁻²⁵ to 10⁻¹⁵ J
            scale: Scale::Linear,  // Already log
            label: "log₁₀(ΔH)".to_string(),
            units: "[J]".to_string(),
        },
        y_axis: AxisConfig {
            variable: Variable::LogDeltaS,
            range: (-25.0, -20.0),  // 10⁻²⁵ to 10⁻²⁰ J/K
            scale: Scale::Linear,
            label: "log₁₀(ΔS)".to_string(),
            units: "[J/K]".to_string(),
        },
        colormap: ColormapConfig {
            name: Colormap::Viridis,
            scale: ColorScale::Log,
            range: (0.01, 100.0),
        },
        contours: ContourConfig {
            levels: vec![0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            color: RGB(0, 0, 0),
            line_width: 0.5,
            show_labels: true,
        },
        annotations: vec![
            Annotation {
                position: (-16.0, -21.0),
                text: "Maximum Information\nRegime".to_string(),
                arrow_to: Some((-15.5, -20.5)),
                style: AnnotationStyle::Peak,
            },
            Annotation {
                position: (-23.0, -24.0),
                text: "Noise Floor".to_string(),
                arrow_to: Some((-24.0, -24.5)),
                style: AnnotationStyle::Basin,
            },
        ],
        output: OutputConfig {
            path: "figures/fig1b_k_landscape_core.pdf".to_string(),
            format: OutputFormat::PDF,
            dpi: 600,
            width: 1200,
            height: 1000,
        },
    };

    let generator = LandscapeGenerator::new();
    generator.generate(&config)?;

    println!("Generated: fig1b_k_landscape_core.pdf");
    Ok(())
}

/// Generate Figure 2b: Gravitational K-Parameter Landscape
pub fn generate_fig2b_gravity_landscape() -> Result<(), Box<dyn Error>> {
    // Similar structure, variant = Gravity
    // ...
}

/// Generate Figure 4b: Dark Sector K-Parameter Landscape
pub fn generate_fig4b_dark_sector_landscape() -> Result<(), Box<dyn Error>> {
    // Similar structure, variant = DarkSector
    // ...
}
```

---

## 🎯 Implementation Timeline

**Phase 1: Core Infrastructure (Days 1-2)**
- Create `k-landscape-generator` crate
- Implement `LandscapeConfig` and data structures
- Add mesh grid generation
- Add basic heatmap plotting with plotters

**Phase 2: Colormap & Contours (Days 3-4)**
- Integrate `colorgrad` for perceptually uniform colormaps
- Implement contour line generation (marching squares)
- Add colorbar rendering
- Test with dummy data

**Phase 3: Variant Implementations (Days 5-6)**
- Implement Core formalism landscape (Variant A)
- Implement Gravity landscape (Variant B)
- Implement Dark Sector landscape (Variant C)
- Verify physics calculations match paper equations

**Phase 4: Annotations & Polish (Day 7)**
- Add annotation system (arrows, text, markers)
- Add special markers (peaks, basins, operating points)
- Implement alt-text generation for accessibility
- Add caption templates

**Phase 5: Export & Integration (Day 8)**
- Add PDF/SVG vector export
- Add high-DPI PNG export
- Integrate into `visualization_generator.rs`
- Update LaTeX document with new figures

**Total**: 8 days for complete implementation

---

## 📖 Documentation Requirements

### Figure Captions (LaTeX)
Complete captions provided above for each variant - ready to copy-paste into `.tex` file

### Alt-Text (Accessibility)
```latex
\begin{figure}
  \centering
  \includegraphics[width=0.85\textwidth]{fig1b_k_landscape_core.pdf}
  \caption{...}
  \label{fig:k_landscape_core}
  \alttext{3D heat-map surface with warm peaks and cool basins representing
  the K-Parameter value across energy uncertainty and entropy variance axes;
  contour lines show iso-K curves. A tall orange peak marks high-K (strong
  quantum fluctuations/long information retention), while deep blue basins
  mark decoherence-dominated regimes.}
\end{figure}
```

### README for k-landscape-generator
Document:
- How to configure each variant
- How to add custom annotations
- How to choose colormaps
- How to export at different resolutions

---

## 🚀 Success Metrics

**Visualization is publication-ready when**:
1. ✅ Colormap is perceptually uniform and color-blind safe
2. ✅ Contour lines are clearly visible against background
3. ✅ Annotations don't overlap with data
4. ✅ Colorbar shows correct units and range
5. ✅ Axes labels include variable names + units
6. ✅ Export as PDF renders correctly at 600 DPI
7. ✅ Caption alone allows reader to understand figure
8. ✅ Alt-text provides accessible description
9. ✅ Reviewers can immediately identify peaks, ridges, basins
10. ✅ Physical interpretation is obvious from visual features

---

## 💡 Future Enhancements (Post-Publication)

1. **Interactive 3D viewer**: WebGL-based interactive landscapes for supplementary materials
2. **Animation**: Show time evolution of landscape (e.g., DM modulation over 1 year)
3. **Multi-variant overlay**: Show all three variants in a single 3×1 panel figure
4. **Gradient arrows**: Show steepest ascent/descent directions
5. **Uncertainty shading**: Add semi-transparent uncertainty regions from parameter fits

---

This specification provides everything needed to implement publication-quality K-Parameter landscape visualizations. The Rust system will generate figures that are:
- **Pedagogically powerful**: Reviewers grasp K-Parameter space immediately
- **Publication-ready**: 600 DPI, vector format, color-blind safe
- **Scientifically accurate**: Directly computed from paper equations
- **Visually stunning**: Warm peaks, cool basins, clear annotations

Ready to implement! 🎨📊✨
