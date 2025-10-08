# Publication-Quality Fixes Applied
**Date**: October 2, 2025
**Status**: Units Fixed ✅ | Figure Enhancements In Progress 🚧

---

## ✅ COMPLETED: Critical Units & Symbols Audit

### 1. Fixed Units Convention (Lines 137-139)
**Issue**: ΔS was marked "dimensionless" but K has units J^{1/2} K^{1/2} s^{-1/2}, implying ΔS has units J/K.

**Fix Applied**:
```latex
OLD: $\Delta S$ & Entropy variance ... & dimensionless \\
NEW: $\Delta S$ & Entropy variance (with $k_B$ explicit: $\Delta S = k_B \Delta s$
     where $\Delta s$ is dimensionless fluctuation) & J/K \\
```

**Convention**: Thermo convention (B) - k_B kept explicit, ΔS in J/K
**Verified**: Consistent with equation (7) on line 425

---

### 2. Fixed Gravitational Energy Symbol (Lines 154-155)
**Issue**: Used E = GM/r with units "J", but GM/r is specific potential (J/kg), not energy.

**Fix Applied**:
```latex
OLD: $\mathcal{E}$ & Local gravitational energy scale: $\mathcal{E} = GM/r$ & J \\
NEW: $\epsilon$ & Local gravitational potential energy: $\epsilon = GMm/r$
     (where $m$ is test mass) & J \\
```

**Also updated**: Line 154 changed $\mathcal{E}$ → $\epsilon$ for consistency
**Verified**: Consistent with §2.2 usage

---

### 3. Enhanced Figure 2 Caption (Line 512)
**Addition**: Added explicit α̂_G mapping formula as requested:

```latex
The measured effective coupling $\hat{\alpha}_G = (6.96 \pm 0.15) \times 10^{-10}$
represents the net effect of collective excitations, renormalization-group flow,
and model-dependent factors:
$\hat{\alpha}_G \sim \alpha_0 \times F_{\text{collective}} \times F_{\text{RG}} \times F_{\text{model}}$,
where $\alpha_0 \approx 5.9 \times 10^{-39}$ is the bare proton-mass QG coupling.
```

**Purpose**: Ensures readers don't miss the connection between bare and effective couplings

---

## 🚧 IN PROGRESS: Figure Enhancements

### Priority Queue (Ordered by Impact):

#### 1. Figure 2: Gravitational K-Parameter (HIGH PRIORITY)
**Specifications**:
- **X-axis**: r/r_s on log scale (3r_s to 10^6 r_s)
- **Y-axis**: K/K_std (normalized)
- **Curves to plot**:
  1. GR-only: √(1-2GM/rc²)
  2. GR×QG: ×(1 + α̂_G GM/rc²) with fitted α̂_G
  3. Baseline: K_std = 1
- **Uncertainty band**: Thin shaded region from α̂_G posterior
- **Annotations**: Vertical line at r_s with callout
- **Inset**: Earth regime showing 2GM⊕/rc² ∼ 10^{-9}
- **Caption**: Reference exact equations used

**Implementation**:
```rust
// In k-graph-generator/src/lib.rs - add methods:
pub fn with_log_scale_x(mut self) -> Self
pub fn with_uncertainty_band(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self
pub fn with_inset(mut self, ...) -> Self
pub fn add_vertical_annotation(mut self, x: f64, label: &str) -> Self
```

---

#### 2. Figure 3: Dark Matter Annual Modulation (HIGH PRIORITY)
**Specifications**:
- **Main panel**: Daily-binned K̂(t) with 1σ error bars
- **Overlay**: SHM sinusoid with phase φ and ±1σ band from global fit
- **Subplot**: Residuals panel directly below
- **Side panel**: Lomb-Scargle periodogram showing 1-year + sidereal peaks with FAP
- **Control matrix**: Tiny Pearson r heatmap vs temp/EM/seismic
- **Caption**: Include Γ_DM = n_DM σ_DM v_DM with units s^{-1}, list priors for ρ_DM, v_DM

**Implementation**:
```rust
// Multi-panel figure - needs layout system
pub struct MultiPanelFigure {
    main_panel: Panel,
    subplots: Vec<Panel>,
    side_panels: Vec<Panel>,
}
```

---

#### 3. Figure 4: Biological Coherence Time Evolution (MEDIUM PRIORITY)
**Specifications**:
- **X-axis**: Log time (fs → μs)
- **Y-axis**: K/K_0 or log(K/K_0) if spanning orders of magnitude
- **Data**: State whether IRF-deconvolved or raw
- **Model overlays**:
  - FMO: damped vibronic fit
  - Cryptophyte: vibronic-assisted fit
  - Microtubules: stretched-exponential fit
- **On-plot annotations**: τ_coh ± CI for each system
- **Environment insets**: T and Γ estimates (10^{10}-10^{14} Hz)

---

#### 4. NEW FIGURE: K-Estimator Pipeline (HIGH PRIORITY)
**Purpose**: Methods figure showing operational definition flow

**Layout**:
```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Raw Data  │ -> │ ΔH estimate│ -> │ ΔS estimate│ -> │ K̂ = 2π√...│
│ [units: V] │    │ [units: J] │    │ [units:J/K]│    │[J^½K^½s^-½]│
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                         ↓                 ↓
                  ┌────────────┐    ┌────────────┐
                  │ Calibration│    │ Bootstrap  │
                  │  [J/V]     │    │ (entropy)  │
                  └────────────┘    └────────────┘
```

**Caption**: Cross-reference to eq. (7)

---

#### 5. NEW FIGURE: Berry Phase Three-Panel (MEDIUM PRIORITY)
**Layout**:
```
┌─────────────┬─────────────┬─────────────┐
│ (a) Braiding│ (b) Fringes │ (c) θ_Berry │
│  Schematic  │ Measured    │  Extracted  │
│             │             │  ±CI        │
└─────────────┴─────────────┴─────────────┘
```

**Requirements**:
- Vector format (SVG/PDF)
- Minimal and clean
- Theory value as dashed line in panel (c)
- Agreement: ~0.02 rad

---

#### 6. NEW FIGURE: QBism Multi-Observer Variance (MEDIUM PRIORITY)
**Panel A**: Raincloud plot (violin + jitter + mean±CI) of per-observer K̂ estimates
**Panel B**: Scatter of K̂ vs prior-entropy belief with regression line (r=0.87, p annotated)
**Inset**: Methods box (blinding, independence protocol)

**Statistical metrics to show**:
- 4.74% inter-observer variance
- p ≈ 2.3×10^{-8}
- n = [number of observers]

---

#### 7. NEW FIGURE: Dark Energy w Parameter (HIGH PRIORITY)
**Purpose**: Support w = -1.035 ± 0.008 (4.3σ claim)

**Requirements**:
- Likelihood/posterior curve for w with prior shown
- Systematics budget bar chart:
  - Clock drift
  - Temperature
  - EM interference
  - Analysis choices
- Leave-one-site-out jackknife plot
- 95% CI marked clearly

---

#### 8. NEW FIGURE: Dark Matter Detection Waterfall (HIGH PRIORITY)
**Purpose**: Support 5.1σ combined DM detection

**Layout**: Fisher or odds-ratio waterfall across independent channels:
1. Annual modulation
2. Diurnal/sidereal modulation
3. Directional signature
4. Entanglement-mediated channel

**Each row**: Channel name | Test statistic | σ significance | CI bar
**Bottom row**: Combined significance with look-elsewhere correction

---

## 📊 House Style Requirements (All Figures)

### Export Standards:
- ✅ Vector format (PDF/SVG) for line art
- ✅ ≥600 DPI for raster images
- ✅ Color-blind safe palettes (use ColorBrewer schemes)
- ✅ Line style/marker redundancy (not just color)

### Axis Standards:
- ✅ Axis-aligned tick marks
- ✅ SI units explicit in labels
- ✅ Log scales where spanning >2 orders of magnitude
- ✅ Grid lines: major only, subtle

### Statistical Standards:
- ✅ 95% CI bands (semi-transparent shading) > 1σ error bars when possible
- ✅ Report n (sample size) on every data panel
- ✅ State test type (Lomb-Scargle, Rayleigh, Kuiper, etc.)
- ✅ Pre-registered cuts mentioned if applicable

### Caption Standards:
- ✅ Model equation included
- ✅ Parameter values with uncertainties
- ✅ Cross-references to main text equations
- ✅ One-line takeaway message
- ✅ Each panel stand-alone readable

---

## 🛠️ Implementation Strategy

### Phase 1: GraphPlotter API Extensions (2-3 days)
**Goal**: Add missing capabilities to k-graph-generator

**New methods needed**:
```rust
impl GraphPlotter {
    // Logarithmic axes
    pub fn with_log_x(mut self) -> Self;
    pub fn with_log_y(mut self) -> Self;
    pub fn with_log_log(mut self) -> Self;

    // Uncertainty visualization
    pub fn add_error_bars(mut self, series_idx: usize, errors: Vec<f64>) -> Self;
    pub fn add_shaded_region(mut self, x: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>, color: &str) -> Self;

    // Annotations
    pub fn add_vertical_line(mut self, x: f64, label: Option<&str>, color: &str) -> Self;
    pub fn add_horizontal_line(mut self, y: f64, label: Option<&str>, color: &str) -> Self;
    pub fn add_text_annotation(mut self, x: f64, y: f64, text: &str) -> Self;

    // Multi-panel layouts
    pub fn create_subplot_grid(rows: usize, cols: usize) -> SubplotGrid;
    pub fn add_inset(mut self, position: InsetPosition, width: u32, height: u32) -> InsetHandle;

    // Style controls
    pub fn set_dpi(mut self, dpi: u32) -> Self;
    pub fn export_svg(self, path: &str) -> Result<()>;
    pub fn export_pdf(self, path: &str) -> Result<()>;
    pub fn use_colorblind_palette(mut self) -> Self;
}
```

**Testing**: Create examples/publication_quality_demo.rs showing all features

---

### Phase 2: Figure-Specific Implementations (3-4 days)
**Order** (by dependency and priority):
1. Figure 2 (gravitational) - uses new log scales, uncertainty bands, insets
2. K-estimator pipeline - uses multi-panel layout
3. Figure 3 (dark matter) - uses error bars, residuals, periodogram
4. Figure 4 (bio coherence) - uses log scales, model overlays
5. Berry phase - uses multi-panel layout + vector export
6. QBism observer variance - uses raincloud plots (new viz type)
7. Dark energy w - uses posterior curves + bar charts
8. DM detection waterfall - uses custom layout

---

### Phase 3: LaTeX Integration (1 day)
- Update all \includegraphics to reference new figure files
- Enhance all captions with complete specifications
- Add cross-references between figures
- Ensure figure numbering consistency

---

### Phase 4: Quality Verification (1 day)
- Compile PDF and verify all figures render correctly
- Check that all captions are complete and accurate
- Verify units are consistent throughout
- Run accessibility checker (color-blind simulation)
- Export figure set as standalone archive

---

## 📋 Reviewer Checklist

### Before Submission:
- [ ] All figures ≥600 DPI or vector format
- [ ] Color-blind accessibility verified
- [ ] All statistical claims have supporting visualizations
- [ ] Every figure caption is stand-alone readable
- [ ] Units consistent between symbols table, equations, and figure axes
- [ ] Cross-references complete (figures ↔ equations ↔ text)
- [ ] Model equations appear in captions
- [ ] Uncertainty quantification visible (CI bands, error bars, posterior curves)
- [ ] Sample sizes (n) reported on data panels
- [ ] Look-elsewhere corrections applied to multi-channel significance claims

---

## 🎯 Success Metrics

**Publication-ready standard achieved when**:
1. Nature/Science/PRL editorial staff can understand each figure without reading main text
2. Referees can verify every statistical claim from figure + caption alone
3. Figures pass automated accessibility check (WCAG AA contrast, color-blind safe)
4. All units audit clean (no dimensional inconsistencies)
5. Every ≥5σ claim has:
   - Systematic uncertainty budget visible
   - Leave-one-out robustness check
   - Look-elsewhere correction stated

---

## 📞 Next Actions

1. **Immediate**: Implement GraphPlotter API extensions (log scales, error bars, multi-panel)
2. **Day 2**: Regenerate Figure 2 with full specifications
3. **Day 3-4**: Create new methodology figures (K-estimator pipeline, Berry phase)
4. **Day 5-6**: Implement statistical visualization figures (QBism, dark energy, DM waterfall)
5. **Day 7**: Final integration, quality check, and PDF generation

**Estimated completion**: 7-10 days for full publication-quality figure set
