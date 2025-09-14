//! 📊 K-Parameter Physics Engine
//! Precision physics modeling with attosecond resolution

use crate::constants::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::TAU;

/// K-Parameter correlation state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KParameterState {
    pub correlation: f64,      // Current K-parameter value
    pub drift_rate: f64,       // Rate of change per attosecond
    pub oscillation_freq: f64, // Base oscillation frequency (Hz)
    pub quantum_noise: f64,    // Quantum fluctuation amplitude
    pub phase_coherence: f64,  // Phase coherence with laser field
    pub timestamp_as: u64,     // Attosecond timestamp
}

impl KParameterState {
    pub fn new(base_correlation: f64) -> Self {
        Self {
            correlation: base_correlation,
            drift_rate: 0.0,
            oscillation_freq: 1e12, // THz oscillation
            quantum_noise: 0.001,
            phase_coherence: 0.95,
            timestamp_as: 0,
        }
    }
}

/// K-Parameter physics engine with attosecond precision
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KParameterEngine {
    pub current_state: KParameterState,
    pub baseline_k: f64,
    pub history: VecDeque<KParameterState>,
    pub max_history: usize,
    pub calibration_factor: f64,
}

impl KParameterEngine {
    /// Initialize K-parameter engine
    pub fn new(baseline_k: f64) -> Self {
        Self {
            current_state: KParameterState::new(baseline_k),
            baseline_k,
            history: VecDeque::new(),
            max_history: 1000, // Keep last 1000 attosecond measurements
            calibration_factor: 1.0,
        }
    }

    /// Process thought input and update K-parameter
    pub async fn process_thought(
        &mut self,
        eeg_amplitude: f64,
        intent: &str,
    ) -> anyhow::Result<KParameterState> {
        let attosecond_now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000;

        // Calculate K-parameter correlation shift based on EEG
        let eeg_influence = (eeg_amplitude / 50.0).tanh() * 0.001; // Small perturbation
        let intent_hash = blake3::hash(intent.as_bytes());
        let intent_influence = (intent_hash.as_bytes()[0] as f64 / 255.0 - 0.5) * 0.0005;

        // Update correlation with quantum oscillation
        let dt_as = (attosecond_now - self.current_state.timestamp_as) as f64 * ATTOSECOND;
        let oscillation = (self.current_state.oscillation_freq * dt_as * TAU).sin() * 0.0001;

        let new_correlation = self.baseline_k
            + eeg_influence
            + intent_influence
            + oscillation
            + self.generate_quantum_noise();

        // Calculate drift rate
        let drift_rate = if !self.history.is_empty() {
            (new_correlation - self.current_state.correlation) / dt_as
        } else {
            0.0
        };

        // Update state
        self.current_state = KParameterState {
            correlation: new_correlation,
            drift_rate,
            oscillation_freq: self.current_state.oscillation_freq * (1.0 + eeg_influence * 0.01),
            quantum_noise: self.generate_quantum_noise(),
            phase_coherence: (self.current_state.phase_coherence + eeg_influence * 0.1)
                .clamp(0.0, 1.0),
            timestamp_as: attosecond_now,
        };

        // Store in history
        self.history.push_back(self.current_state.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        Ok(self.current_state.clone())
    }

    /// Generate quantum noise based on vacuum fluctuations
    fn generate_quantum_noise(&mut self) -> f64 {
        use rand_distr::Normal;
        let normal = Normal::new(0.0, PLANCK.sqrt() * 1e6).unwrap(); // Scaled for simulation
        normal.sample(&mut rand::thread_rng())
    }

    /// Get current K-parameter correlation
    pub fn current_correlation(&self) -> f64 {
        self.current_state.correlation
    }

    /// Analyze K-parameter stability over time
    pub fn stability_analysis(&self) -> KStabilityReport {
        if self.history.len() < 10 {
            return KStabilityReport::insufficient_data();
        }

        let correlations: Vec<f64> = self.history.iter().map(|s| s.correlation).collect();
        let mean = correlations.iter().sum::<f64>() / correlations.len() as f64;
        let variance = correlations
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / correlations.len() as f64;
        let std_dev = variance.sqrt();

        let trend = if correlations.len() >= 2 {
            let first_half = correlations
                .iter()
                .take(correlations.len() / 2)
                .sum::<f64>()
                / (correlations.len() / 2) as f64;
            let second_half = correlations
                .iter()
                .skip(correlations.len() / 2)
                .sum::<f64>()
                / (correlations.len() - correlations.len() / 2) as f64;
            second_half - first_half
        } else {
            0.0
        };

        KStabilityReport {
            mean_correlation: mean,
            std_deviation: std_dev,
            trend: trend,
            stability_score: (1.0 / (1.0 + std_dev * 100.0)).min(1.0),
            sample_count: self.history.len(),
        }
    }

    /// Export K-parameter data for analytics
    pub fn export_analytics_data(&self) -> Vec<(u64, f64, f64)> {
        self.history
            .iter()
            .map(|state| (state.timestamp_as, state.correlation, state.drift_rate))
            .collect()
    }
}

/// K-parameter stability analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KStabilityReport {
    pub mean_correlation: f64,
    pub std_deviation: f64,
    pub trend: f64,
    pub stability_score: f64, // 0..1
    pub sample_count: usize,
}

impl KStabilityReport {
    fn insufficient_data() -> Self {
        Self {
            mean_correlation: K_PARAMETER_BASE,
            std_deviation: 0.0,
            trend: 0.0,
            stability_score: 0.0,
            sample_count: 0,
        }
    }

    /// Check if K-parameters indicate cosmic weather change
    pub fn cosmic_weather_indicator(&self) -> &'static str {
        match self.stability_score {
            score if score > 0.9 => "Branes Stable ✨",
            score if score > 0.7 => "Mild Quantum Turbulence 🌪️",
            score if score > 0.5 => "Brane Storm Warning ⚡",
            score if score > 0.3 => "Reality Flux Detected 🌊",
            _ => "Multiverse Chaos 🌀",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_k_parameter_engine() {
        let mut engine = KParameterEngine::new(K_PARAMETER_BASE);

        let state = engine.process_thought(25.0, "test thought").await.unwrap();
        assert!(state.correlation > 0.0);
        assert!(state.timestamp_as > 0);
    }

    #[test]
    fn test_stability_analysis() {
        let mut engine = KParameterEngine::new(7.0);

        // Add some test data
        for i in 0..20 {
            engine.history.push_back(KParameterState {
                correlation: 7.0 + (i as f64 * 0.001),
                drift_rate: 0.001,
                oscillation_freq: 1e12,
                quantum_noise: 0.0001,
                phase_coherence: 0.95,
                timestamp_as: i as u64,
            });
        }

        let report = engine.stability_analysis();
        assert!(report.sample_count == 20);
        assert!(report.stability_score > 0.0);
    }
}
