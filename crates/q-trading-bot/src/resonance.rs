/// Breit-Wigner resonance gate for DEX swap timing.
///
/// Adapted from particle physics: σ(E) = σ_peak × Γ²/[(E-M)² + Γ²/4]
/// Here: η(R) = Γ²/[(R-R₀)² + Γ²/4]  where R = reserve_in/reserve_out
///
/// A balanced pool (R = R₀ = 1.0) is the resonance peak — minimum slippage.
/// Only execute swaps when η exceeds the configured threshold.

/// Pool balance ratio and resonance efficiency.
#[derive(Debug, Clone)]
pub struct ResonanceState {
    /// Current ratio R = reserve_in / reserve_out
    pub ratio: f64,
    /// Resonance efficiency η ∈ [0, 1]
    pub efficiency: f64,
    /// Natural linewidth Γ (pool "width" — higher TVL = wider resonance)
    pub gamma: f64,
}

impl ResonanceState {
    pub fn above_threshold(&self, threshold: f64) -> bool {
        self.efficiency >= threshold
    }
}

/// Compute resonance efficiency for a pool at its current reserve state.
///
/// # Arguments
/// * `reserve_in`  - raw u128 reserves for the input token (24-decimal)
/// * `reserve_out` - raw u128 reserves for the output token (24-decimal)
/// * `gamma`       - linewidth parameter; use `DEFAULT_GAMMA` if unsure
pub fn compute_resonance(reserve_in: u128, reserve_out: u128, gamma: f64) -> ResonanceState {
    if reserve_in == 0 || reserve_out == 0 {
        return ResonanceState { ratio: 0.0, efficiency: 0.0, gamma };
    }

    let r = reserve_in as f64 / reserve_out as f64;
    let r0 = 1.0_f64; // resonance centre — perfectly balanced pool

    // Breit-Wigner: η = Γ² / [(R-R₀)² + Γ²/4]
    let gamma_sq = gamma * gamma;
    let denom = (r - r0).powi(2) + gamma_sq / 4.0;
    let raw_eta = gamma_sq / denom;

    // Normalise to [0, 1]: maximum occurs at R = R₀ → η_max = Γ²/(Γ²/4) = 4
    let eta = (raw_eta / 4.0).clamp(0.0, 1.0);

    ResonanceState { ratio: r, efficiency: eta, gamma }
}

/// Derive gamma automatically from pool TVL (deeper pools = wider resonance).
/// Γ = min(1.0, sqrt(tvl_display) / 1000)
pub fn gamma_from_tvl(tvl_display: f64) -> f64 {
    (tvl_display.sqrt() / 1000.0).clamp(0.05, 1.0)
}

/// FCC-ee operating mode — maps to DEX strategy intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FccMode {
    /// Z-pole (91.2 GeV) → high-frequency DCA on deep pools. Threshold = 0.80
    ZPole,
    /// WW threshold (160 GeV) → watch for phase transitions. Threshold = 0.60
    WwThreshold,
    /// ZH production (240 GeV) → accumulate on new pools. Threshold = 0.40
    ZhProduction,
    /// tt̄ threshold (365 GeV) → surgical swaps on thin pools only. Threshold = 0.95
    TtBar,
}

impl FccMode {
    pub fn resonance_threshold(self) -> f64 {
        match self {
            FccMode::ZPole       => 0.80,
            FccMode::WwThreshold => 0.60,
            FccMode::ZhProduction=> 0.40,
            FccMode::TtBar       => 0.95,
        }
    }

    /// Auto-select mode based on pool TVL (display units).
    pub fn from_pool_depth(tvl: f64) -> Self {
        if tvl > 100_000.0 { FccMode::ZPole }
        else if tvl > 10_000.0 { FccMode::WwThreshold }
        else if tvl > 1_000.0 { FccMode::ZhProduction }
        else { FccMode::TtBar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_pool_is_resonance_peak() {
        // Equal reserves → R = 1.0 → η should be 1.0
        let res = compute_resonance(1_000_000, 1_000_000, 0.5);
        assert!((res.efficiency - 1.0).abs() < 0.01);
    }

    #[test]
    fn imbalanced_pool_lower_efficiency() {
        let balanced = compute_resonance(1_000_000, 1_000_000, 0.5);
        let imbalanced = compute_resonance(5_000_000, 1_000_000, 0.5);
        assert!(imbalanced.efficiency < balanced.efficiency);
    }
}
