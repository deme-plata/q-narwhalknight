//! 🎯 K-Parameter Consensus Metrics
//!
//! Comprehensive metrics and monitoring for K-Parameter quantum phase analysis

use crate::{PhaseAnalysis, PhaseTransition, PhaseRecommendation};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json;

/// 🎯 K-Parameter Consensus Metrics
///
/// Complete monitoring data for K-Parameter phase analysis,
/// exportable to Prometheus, Grafana, or custom dashboards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KParameterMetrics {
    /// Current K-Parameter value
    pub current_k: f64,

    /// Historical K values (last N rounds)
    pub k_trend: Vec<f64>,

    /// Energy variance (ΔH)
    pub energy_variance: f64,

    /// Entropy variance (Δs)
    pub entropy_variance: f64,

    /// Round duration (τ) in seconds
    pub round_duration: f64,

    /// Phase stability metric [0, 1]
    pub phase_stability: f64,

    /// Transition risk assessment [0, 1]
    pub transition_risk: f64,

    /// Current phase state
    pub phase_state: PhaseTransition,

    /// Operational recommendations
    pub recommendations: Vec<String>,

    /// Timestamp of last update
    pub last_update: u64,

    /// Total rounds analyzed
    pub total_rounds: u64,

    /// Number of phase transitions detected
    pub transition_count: u64,

    /// Number of critical events
    pub critical_events: u64,
}

impl Default for KParameterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl KParameterMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            current_k: 0.0,
            k_trend: Vec::new(),
            energy_variance: 0.0,
            entropy_variance: 0.0,
            round_duration: 0.0,
            phase_stability: 1.0,
            transition_risk: 0.0,
            phase_state: PhaseTransition::Stable,
            recommendations: Vec::new(),
            last_update: Self::current_timestamp(),
            total_rounds: 0,
            transition_count: 0,
            critical_events: 0,
        }
    }

    /// 🎯 Update metrics with new round data
    pub fn update(&mut self, analysis: &PhaseAnalysis) {
        self.current_k = analysis.k_parameter;
        self.k_trend.push(analysis.k_parameter);
        self.energy_variance = analysis.energy_variance;
        self.entropy_variance = analysis.entropy_variance;
        self.round_duration = analysis.round_duration;
        self.phase_stability = analysis.stability;
        self.phase_state = analysis.phase_transition;
        self.last_update = Self::current_timestamp();
        self.total_rounds += 1;

        // Track transitions
        if matches!(analysis.phase_transition, PhaseTransition::Approaching | PhaseTransition::Critical) {
            self.transition_count += 1;
        }

        // Track critical events
        if analysis.needs_emergency_action() {
            self.critical_events += 1;
        }

        // Compute derived metrics
        self.transition_risk = self.compute_transition_risk();
        self.recommendations = self.generate_recommendations(analysis);

        // Trim trend if too long (keep last 100)
        if self.k_trend.len() > 100 {
            self.k_trend.remove(0);
        }
    }

    /// 🎯 Compute transition risk based on K-Parameter behavior
    ///
    /// Risk is assessed by:
    /// - K magnitude (higher K = higher risk)
    /// - K rate of change (faster change = higher risk)
    /// - Historical volatility (more volatile = higher risk)
    fn compute_transition_risk(&self) -> f64 {
        if self.k_trend.len() < 3 {
            return 0.0;
        }

        let recent: Vec<f64> = self.k_trend.iter().rev().take(3).cloned().collect();

        // Compute rate of change
        let changes: Vec<f64> = recent.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .collect();

        let avg_change: f64 = if changes.is_empty() {
            0.0
        } else {
            changes.iter().sum::<f64>() / changes.len() as f64
        };

        // Risk increases with K magnitude and change rate
        let magnitude_risk = (self.current_k / 10.0).min(1.0);
        let change_risk = (avg_change / 2.0).min(1.0);

        // Combined risk (weighted average)
        (0.6 * magnitude_risk + 0.4 * change_risk).min(1.0)
    }

    /// 🎯 Generate human-readable recommendations
    fn generate_recommendations(&self, analysis: &PhaseAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        match analysis.phase_transition {
            PhaseTransition::Stable => {
                recommendations.push("✅ System stable - normal operation".to_string());

                if self.current_k < 0.1 {
                    recommendations.push("💡 K very low - consider optimizing convergence speed".to_string());
                }
            }
            PhaseTransition::Approaching => {
                recommendations.push("⚠️  Monitor closely - phase transition approaching".to_string());
                recommendations.push("📊 Consider increasing Byzantine detection sensitivity".to_string());
                recommendations.push("🔍 Review recent network changes".to_string());
            }
            PhaseTransition::Critical => {
                recommendations.push("🚨 CRITICAL: Phase transition detected".to_string());
                recommendations.push("🛡️  Activate emergency consensus protocols".to_string());
                recommendations.push("📢 Alert network operators immediately".to_string());
                recommendations.push("💾 Save diagnostic data for analysis".to_string());
            }
        }

        if self.transition_risk > 0.7 {
            recommendations.push("⚡ High transition risk - prepare contingency plans".to_string());
        }

        if self.phase_stability < 0.5 {
            recommendations.push("📉 Low stability detected - investigate network conditions".to_string());
        }

        if self.energy_variance > 10.0 {
            recommendations.push("⚡ High energy variance - possible Byzantine activity".to_string());
        }

        if self.entropy_variance > 5.0 {
            recommendations.push("🌀 High entropy variance - check node synchronization".to_string());
        }

        recommendations
    }

    /// 🎯 Export metrics for Prometheus
    pub fn export_prometheus_metrics(&self) -> String {
        format!(
            "# HELP resonance_k_parameter Current K-Parameter value\n\
             # TYPE resonance_k_parameter gauge\n\
             resonance_k_parameter {}\n\
             \n\
             # HELP resonance_energy_variance Energy variance (ΔH)\n\
             # TYPE resonance_energy_variance gauge\n\
             resonance_energy_variance {}\n\
             \n\
             # HELP resonance_entropy_variance Entropy variance (Δs)\n\
             # TYPE resonance_entropy_variance gauge\n\
             resonance_entropy_variance {}\n\
             \n\
             # HELP resonance_phase_stability Phase stability metric\n\
             # TYPE resonance_phase_stability gauge\n\
             resonance_phase_stability {}\n\
             \n\
             # HELP resonance_transition_risk Transition risk assessment\n\
             # TYPE resonance_transition_risk gauge\n\
             resonance_transition_risk {}\n\
             \n\
             # HELP resonance_round_duration Round duration in seconds (τ)\n\
             # TYPE resonance_round_duration gauge\n\
             resonance_round_duration {}\n\
             \n\
             # HELP resonance_total_rounds Total rounds analyzed\n\
             # TYPE resonance_total_rounds counter\n\
             resonance_total_rounds {}\n\
             \n\
             # HELP resonance_transition_count Number of phase transitions detected\n\
             # TYPE resonance_transition_count counter\n\
             resonance_transition_count {}\n\
             \n\
             # HELP resonance_critical_events Number of critical events\n\
             # TYPE resonance_critical_events counter\n\
             resonance_critical_events {}\n",
            self.current_k,
            self.energy_variance,
            self.entropy_variance,
            self.phase_stability,
            self.transition_risk,
            self.round_duration,
            self.total_rounds,
            self.transition_count,
            self.critical_events
        )
    }

    /// 🎯 Export as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// 🎯 Get summary string for logging
    pub fn summary(&self) -> String {
        format!(
            "K-Parameter: K={:.3} | Phase={} | Stability={:.1}% | Risk={:.1}% | Round={}",
            self.current_k,
            self.phase_state,
            self.phase_stability * 100.0,
            self.transition_risk * 100.0,
            self.total_rounds
        )
    }

    /// 🎯 Get detailed report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔════════════════════════════════════════════════════════════╗\n");
        report.push_str("║         K-PARAMETER QUANTUM PHASE ANALYSIS REPORT          ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════╝\n\n");

        report.push_str(&format!("🎯 K-Parameter: {:.4}\n", self.current_k));
        report.push_str(&format!("📊 Phase State: {}\n", self.phase_state));
        report.push_str(&format!("📈 Stability:   {:.1}%\n", self.phase_stability * 100.0));
        report.push_str(&format!("⚠️  Risk Level:  {:.1}%\n\n", self.transition_risk * 100.0));

        report.push_str("📐 Quantum Parameters:\n");
        report.push_str(&format!("   ΔH (Energy Variance):  {:.4}\n", self.energy_variance));
        report.push_str(&format!("   Δs (Entropy Variance): {:.4}\n", self.entropy_variance));
        report.push_str(&format!("   τ (Round Duration):    {:.3}s\n\n", self.round_duration));

        report.push_str("📊 Statistics:\n");
        report.push_str(&format!("   Total Rounds:          {}\n", self.total_rounds));
        report.push_str(&format!("   Transitions Detected:  {}\n", self.transition_count));
        report.push_str(&format!("   Critical Events:       {}\n\n", self.critical_events));

        if !self.recommendations.is_empty() {
            report.push_str("💡 Recommendations:\n");
            for rec in &self.recommendations {
                report.push_str(&format!("   {}\n", rec));
            }
        }

        report
    }

    /// Get current Unix timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Check if metrics indicate healthy consensus
    pub fn is_healthy(&self) -> bool {
        self.phase_stability > 0.7
            && self.transition_risk < 0.5
            && matches!(self.phase_state, PhaseTransition::Stable)
    }

    /// Get K-Parameter trend direction
    pub fn k_trend_direction(&self) -> TrendDirection {
        if self.k_trend.len() < 2 {
            return TrendDirection::Stable;
        }

        let recent: Vec<f64> = self.k_trend.iter().rev().take(5).cloned().collect();
        if recent.len() < 2 {
            return TrendDirection::Stable;
        }

        let first = recent.last().unwrap();
        let last = recent.first().unwrap();
        let change = last - first;

        if change.abs() < 0.1 {
            TrendDirection::Stable
        } else if change > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        }
    }
}

/// Trend direction for K-Parameter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Increasing => write!(f, "↗ Increasing"),
            Self::Decreasing => write!(f, "↘ Decreasing"),
            Self::Stable => write!(f, "→ Stable"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhaseTransition, PhaseRecommendation, ConsensusTuning};

    fn create_test_analysis(k: f64, phase: PhaseTransition) -> PhaseAnalysis {
        PhaseAnalysis {
            k_parameter: k,
            energy_variance: 1.0,
            entropy_variance: 1.0,
            round_duration: 1.0,
            phase_transition: phase,
            stability: 0.9,
            recommendation: PhaseRecommendation::NormalOperation,
            tuning_applied: ConsensusTuning {
                learning_rate: 0.1,
                max_iterations: 500,
                spectral_threshold: 0.1,
                convergence_tolerance: 1e-6,
            },
        }
    }

    #[test]
    fn test_metrics_update() {
        let mut metrics = KParameterMetrics::new();
        let analysis = create_test_analysis(0.5, PhaseTransition::Stable);

        metrics.update(&analysis);

        assert_eq!(metrics.current_k, 0.5);
        assert_eq!(metrics.total_rounds, 1);
        assert!(metrics.k_trend.contains(&0.5));
    }

    #[test]
    fn test_transition_risk_computation() {
        let mut metrics = KParameterMetrics::new();

        // Add stable K values
        metrics.k_trend = vec![1.0, 1.1, 0.9, 1.0];
        metrics.current_k = 1.0;
        let low_risk = metrics.compute_transition_risk();
        assert!(low_risk < 0.3);

        // Add volatile K values
        metrics.k_trend = vec![1.0, 5.0, 2.0, 8.0];
        metrics.current_k = 8.0;
        let high_risk = metrics.compute_transition_risk();
        assert!(high_risk > 0.5);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = KParameterMetrics::new();
        let prometheus_output = metrics.export_prometheus_metrics();

        assert!(prometheus_output.contains("resonance_k_parameter"));
        assert!(prometheus_output.contains("resonance_energy_variance"));
        assert!(prometheus_output.contains("resonance_entropy_variance"));
    }

    #[test]
    fn test_health_check() {
        let mut metrics = KParameterMetrics::new();

        // Healthy state
        metrics.phase_stability = 0.9;
        metrics.transition_risk = 0.2;
        metrics.phase_state = PhaseTransition::Stable;
        assert!(metrics.is_healthy());

        // Unhealthy state
        metrics.phase_stability = 0.4;
        metrics.transition_risk = 0.8;
        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_trend_direction() {
        let mut metrics = KParameterMetrics::new();

        // Increasing trend
        metrics.k_trend = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        assert_eq!(metrics.k_trend_direction(), TrendDirection::Increasing);

        // Decreasing trend
        metrics.k_trend = vec![3.0, 2.5, 2.0, 1.5, 1.0];
        assert_eq!(metrics.k_trend_direction(), TrendDirection::Decreasing);

        // Stable trend
        metrics.k_trend = vec![1.0, 1.05, 0.95, 1.0, 1.02];
        assert_eq!(metrics.k_trend_direction(), TrendDirection::Stable);
    }
}
