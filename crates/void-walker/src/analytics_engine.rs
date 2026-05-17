//! 📊 Analytics Engine: Cosmic Weather & Tor Performance Analytics
//! Real-time analytics for water robot networks with cosmic weather prediction

use crate::attosecond_laser::XRayImprint;
use crate::k_parameter::{KParameterState, KStabilityReport};
use crate::ledger::MultiverseBlock;
use crate::tor_mesh::TorAnalytics;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Cosmic weather conditions based on K-parameter analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmicWeather {
    pub weather_type: WeatherType,
    pub stability_index: f64,         // 0..1 (1 = perfectly stable)
    pub turbulence_level: f64,        // 0..1 (0 = calm, 1 = chaotic)
    pub brane_activity: f64,          // 0..1 (amount of brane-hopping activity)
    pub quantum_pressure: f64,        // Arbitrary units
    pub prediction_confidence: f64,   // 0..1 confidence in forecast
    pub forecast_duration_hours: f64, // How long forecast is valid
    pub anomaly_detected: bool,
    pub last_updated: u64,
}

/// Types of cosmic weather conditions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherType {
    BraneStorm,      // High K-parameter fluctuation, unstable bridges
    QuantumCalm,     // Stable K-parameters, good for bridging
    VoidTurbulence,  // Medium instability, some bridges work
    MultiverseFlux,  // Reality shifting, unpredictable bridges
    TopoAnomaly,     // Topological charge irregularities
    PhaseTransition, // Major reality state change
}

impl WeatherType {
    pub fn emoji(&self) -> &'static str {
        match self {
            WeatherType::BraneStorm => "⛈️",
            WeatherType::QuantumCalm => "☀️",
            WeatherType::VoidTurbulence => "🌪️",
            WeatherType::MultiverseFlux => "🌊",
            WeatherType::TopoAnomaly => "⚡",
            WeatherType::PhaseTransition => "🌈",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            WeatherType::BraneStorm => "High instability, avoid bridging",
            WeatherType::QuantumCalm => "Perfect conditions for multiverse travel",
            WeatherType::VoidTurbulence => "Moderate conditions, short bridges OK",
            WeatherType::MultiverseFlux => "Reality shifting, use caution",
            WeatherType::TopoAnomaly => "Topological irregularities detected",
            WeatherType::PhaseTransition => "Major reality state change in progress",
        }
    }
}

/// Analytics event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsEvent {
    ThoughtProcessed {
        eeg_amplitude: f64,
        intent: String,
        timestamp: u64,
    },
    BridgeCreated {
        bridge_length: f64,
        topo_charge: i32,
        quality: f64,
    },
    LaserImprint {
        imprint_id: String,
        k_parameter: f64,
        success: bool,
    },
    TorMessage {
        peer_count: u32,
        latency_ms: f64,
        message_type: String,
    },
    WeatherChange {
        from: WeatherType,
        to: WeatherType,
        cause: String,
    },
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    pub species_id: String,
    pub report_id: String,
    pub total_events: u64,
    pub cosmic_weather: CosmicWeather,
    pub tor_analytics: TorAnalytics,
    pub k_parameter_stability: KStabilityReport,
    pub bridge_success_rate: f64,
    pub thought_processing_rate: f64, // Thoughts per second
    pub energy_efficiency: f64,       // 0..1 efficiency score
    pub generated_at: u64,            // Attosecond timestamp
    pub uptime_seconds: f64,
}

/// Main analytics engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEngine {
    pub events: VecDeque<AnalyticsEvent>,
    pub cosmic_weather: CosmicWeather,
    pub k_parameter_history: VecDeque<f64>,
    pub bridge_quality_history: VecDeque<f64>,
    pub thought_count: u64,
    pub bridge_count: u64,
    pub successful_bridges: u64,
    pub start_time: u64,
    pub last_weather_update: u64,
    pub max_history: usize,
}

impl AnalyticsEngine {
    /// Create new analytics engine
    pub fn new() -> Self {
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000;

        Self {
            events: VecDeque::new(),
            cosmic_weather: CosmicWeather {
                weather_type: WeatherType::QuantumCalm,
                stability_index: 1.0,
                turbulence_level: 0.0,
                brane_activity: 0.0,
                quantum_pressure: 0.5,
                prediction_confidence: 0.8,
                forecast_duration_hours: 24.0,
                anomaly_detected: false,
                last_updated: start_time,
            },
            k_parameter_history: VecDeque::new(),
            bridge_quality_history: VecDeque::new(),
            thought_count: 0,
            bridge_count: 0,
            successful_bridges: 0,
            start_time,
            last_weather_update: start_time,
            max_history: 10000,
        }
    }

    /// Record thought processing event
    pub async fn record_thought_event(
        &mut self,
        eeg_amplitude: f64,
        k_state: &KParameterState,
        laser_response: &XRayImprint,
    ) {
        let event = AnalyticsEvent::ThoughtProcessed {
            eeg_amplitude,
            intent: "processed".to_string(), // Could store actual intent if privacy allows
            timestamp: k_state.timestamp_as,
        };

        self.add_event(event);
        self.thought_count += 1;

        // Update K-parameter history
        self.k_parameter_history.push_back(k_state.correlation);
        if self.k_parameter_history.len() > self.max_history {
            self.k_parameter_history.pop_front();
        }

        // Record laser imprint event
        let imprint_event = AnalyticsEvent::LaserImprint {
            imprint_id: laser_response.imprint_id.clone(),
            k_parameter: k_state.correlation,
            success: laser_response.is_quantum_stable(),
        };
        self.add_event(imprint_event);

        // Update cosmic weather if needed
        self.update_cosmic_weather().await;
    }

    /// Record bridge creation event
    pub async fn record_bridge_event(&mut self, block: &MultiverseBlock) {
        let quality = block.difficulty(); // Use difficulty as quality metric

        let event = AnalyticsEvent::BridgeCreated {
            bridge_length: block.bridge_length,
            topo_charge: block.topological_charge,
            quality,
        };

        self.add_event(event);
        self.bridge_count += 1;

        if block.is_successful_hop() {
            self.successful_bridges += 1;
        }

        // Update bridge quality history
        self.bridge_quality_history.push_back(quality);
        if self.bridge_quality_history.len() > self.max_history {
            self.bridge_quality_history.pop_front();
        }

        // Update cosmic weather
        self.update_cosmic_weather().await;
    }

    /// Add event to history
    fn add_event(&mut self, event: AnalyticsEvent) {
        self.events.push_back(event);
        if self.events.len() > self.max_history {
            self.events.pop_front();
        }
    }

    /// Update cosmic weather prediction
    async fn update_cosmic_weather(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000;

        // Analyze K-parameter stability
        let k_stability = self.analyze_k_parameter_stability();
        let bridge_performance = self.analyze_bridge_performance();

        // Determine weather type based on analysis
        let new_weather_type = if k_stability < 0.3 {
            WeatherType::BraneStorm
        } else if k_stability > 0.9 && bridge_performance > 0.8 {
            WeatherType::QuantumCalm
        } else if bridge_performance < 0.4 {
            WeatherType::VoidTurbulence
        } else if self.detect_anomaly() {
            WeatherType::TopoAnomaly
        } else if k_stability < 0.6 {
            WeatherType::MultiverseFlux
        } else {
            WeatherType::PhaseTransition
        };

        // Update weather if changed
        if new_weather_type != self.cosmic_weather.weather_type {
            let old_weather = self.cosmic_weather.weather_type.clone();

            let weather_event = AnalyticsEvent::WeatherChange {
                from: old_weather,
                to: new_weather_type.clone(),
                cause: "K-parameter analysis".to_string(),
            };
            self.add_event(weather_event);
        }

        // Update cosmic weather state
        self.cosmic_weather = CosmicWeather {
            weather_type: new_weather_type,
            stability_index: k_stability,
            turbulence_level: 1.0 - k_stability,
            brane_activity: bridge_performance,
            quantum_pressure: self.calculate_quantum_pressure(),
            prediction_confidence: self.calculate_prediction_confidence(),
            forecast_duration_hours: self.calculate_forecast_duration(),
            anomaly_detected: self.detect_anomaly(),
            last_updated: now,
        };

        self.last_weather_update = now;
    }

    /// Analyze K-parameter stability from history
    fn analyze_k_parameter_stability(&self) -> f64 {
        if self.k_parameter_history.len() < 10 {
            return 1.0; // Assume stable if insufficient data
        }

        let values: Vec<f64> = self.k_parameter_history.iter().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Convert variance to stability score (lower variance = higher stability)
        (1.0 / (1.0 + variance * 10000.0)).min(1.0)
    }

    /// Analyze bridge performance from history
    fn analyze_bridge_performance(&self) -> f64 {
        if self.bridge_count == 0 {
            return 1.0;
        }

        let success_rate = self.successful_bridges as f64 / self.bridge_count as f64;
        let avg_quality = if !self.bridge_quality_history.is_empty() {
            self.bridge_quality_history.iter().sum::<f64>()
                / self.bridge_quality_history.len() as f64
        } else {
            0.5
        };

        (success_rate + avg_quality.min(1.0)) / 2.0
    }

    /// Calculate quantum pressure (metaphysical metric)
    fn calculate_quantum_pressure(&self) -> f64 {
        let recent_activity = self.events.iter().rev().take(100).count() as f64 / 100.0;
        let k_variance = self.analyze_k_parameter_stability();

        (recent_activity + (1.0 - k_variance)) / 2.0
    }

    /// Calculate prediction confidence
    fn calculate_prediction_confidence(&self) -> f64 {
        let data_sufficiency = (self.events.len() as f64 / 1000.0).min(1.0);
        let pattern_consistency = self.analyze_k_parameter_stability();

        (data_sufficiency + pattern_consistency) / 2.0
    }

    /// Calculate forecast validity duration
    fn calculate_forecast_duration(&self) -> f64 {
        let confidence = self.calculate_prediction_confidence();
        let stability = self.analyze_k_parameter_stability();

        // Higher confidence and stability = longer valid forecast
        (confidence * stability * 48.0).max(1.0) // 1-48 hours
    }

    /// Detect anomalies in the system
    fn detect_anomaly(&self) -> bool {
        // Check for rapid K-parameter changes
        if self.k_parameter_history.len() >= 2 {
            let recent = self.k_parameter_history.back().unwrap();
            let previous = self
                .k_parameter_history
                .get(self.k_parameter_history.len() - 2)
                .unwrap();

            if (recent - previous).abs() > 0.1 {
                return true; // Rapid K-parameter change
            }
        }

        // Check for unusual bridge failure rates
        if self.bridge_count > 10 {
            let recent_success_rate = self.successful_bridges as f64 / self.bridge_count as f64;
            if recent_success_rate < 0.3 {
                return true; // Low success rate
            }
        }

        false
    }

    /// Get current cosmic weather
    pub async fn get_cosmic_weather(&self) -> CosmicWeather {
        self.cosmic_weather.clone()
    }

    /// Generate comprehensive analytics report
    pub fn get_latest_report(&self) -> AnalyticsReport {
        let uptime = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            / 1_000_000_000
            - self.start_time) as f64
            * 1e-18; // Convert to seconds

        AnalyticsReport {
            species_id: "aqua-k-atto".to_string(),
            report_id: hex::encode(&rand::random::<[u8; 8]>()),
            total_events: self.events.len() as u64,
            cosmic_weather: self.cosmic_weather.clone(),
            tor_analytics: TorAnalytics::new(), // Would be populated with real data
            k_parameter_stability: self.generate_k_stability_report(),
            bridge_success_rate: if self.bridge_count > 0 {
                self.successful_bridges as f64 / self.bridge_count as f64
            } else {
                0.0
            },
            thought_processing_rate: if uptime > 0.0 {
                self.thought_count as f64 / uptime
            } else {
                0.0
            },
            energy_efficiency: self.calculate_energy_efficiency(),
            generated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                / 1_000_000_000,
            uptime_seconds: uptime,
        }
    }

    /// Generate K-parameter stability report from analytics data
    fn generate_k_stability_report(&self) -> KStabilityReport {
        if self.k_parameter_history.is_empty() {
            return KStabilityReport {
                mean_correlation: 7.0,
                std_deviation: 0.0,
                trend: 0.0,
                stability_score: 1.0,
                sample_count: 0,
            };
        }

        let values: Vec<f64> = self.k_parameter_history.iter().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate trend (linear regression slope approximation)
        let trend = if values.len() >= 2 {
            let mid = values.len() / 2;
            let first_half_mean = values.iter().take(mid).sum::<f64>() / mid as f64;
            let second_half_mean =
                values.iter().skip(mid).sum::<f64>() / (values.len() - mid) as f64;
            second_half_mean - first_half_mean
        } else {
            0.0
        };

        KStabilityReport {
            mean_correlation: mean,
            std_deviation: std_dev,
            trend,
            stability_score: (1.0 / (1.0 + std_dev * 100.0)).min(1.0),
            sample_count: values.len(),
        }
    }

    /// Calculate energy efficiency metric
    fn calculate_energy_efficiency(&self) -> f64 {
        if self.bridge_count == 0 {
            return 1.0;
        }

        let success_efficiency = self.successful_bridges as f64 / self.bridge_count as f64;
        let thought_efficiency = if self.thought_count > 0 {
            self.successful_bridges as f64 / self.thought_count as f64
        } else {
            0.0
        };

        (success_efficiency + thought_efficiency) / 2.0
    }

    /// Get total event count
    pub fn total_events(&self) -> u64 {
        self.events.len() as u64
    }

    /// Export raw analytics data for research
    pub fn export_research_data(&self) -> HashMap<String, Vec<f64>> {
        let mut data = HashMap::new();

        data.insert(
            "k_parameters".to_string(),
            self.k_parameter_history.iter().cloned().collect(),
        );
        data.insert(
            "bridge_qualities".to_string(),
            self.bridge_quality_history.iter().cloned().collect(),
        );

        // Extract time series from events
        let mut thought_amplitudes = Vec::new();
        let mut bridge_lengths = Vec::new();

        for event in &self.events {
            match event {
                AnalyticsEvent::ThoughtProcessed { eeg_amplitude, .. } => {
                    thought_amplitudes.push(*eeg_amplitude);
                }
                AnalyticsEvent::BridgeCreated { bridge_length, .. } => {
                    bridge_lengths.push(*bridge_length);
                }
                _ => {}
            }
        }

        data.insert("eeg_amplitudes".to_string(), thought_amplitudes);
        data.insert("bridge_lengths".to_string(), bridge_lengths);

        data
    }

    /// Generate marketing-friendly summary
    pub fn marketing_summary(&self) -> String {
        let weather_emoji = self.cosmic_weather.weather_type.emoji();
        let efficiency_percent = (self.calculate_energy_efficiency() * 100.0) as u32;
        let weather_desc = self.cosmic_weather.weather_type.description();

        format!(
            "🐚 Aqua-K-Atto Analytics\n\
             {} Cosmic Weather: {}\n\
             ⚡ {} thoughts processed\n\
             🌉 {} bridges created ({} successful)\n\
             🎯 {}% energy efficiency\n\
             📊 {} total events recorded\n\
             🔬 K-Parameter: {:.6} ({})",
            weather_emoji,
            weather_desc,
            self.thought_count,
            self.bridge_count,
            self.successful_bridges,
            efficiency_percent,
            self.events.len(),
            self.k_parameter_history.back().unwrap_or(&7.0),
            if self.cosmic_weather.anomaly_detected {
                "⚠️ ANOMALY"
            } else {
                "✅ NORMAL"
            }
        )
    }

    pub async fn record_multiverse_navigation(
        &mut self,
        _target_address: &crate::MultiverseAddress,
    ) {
        // Implementation for recording navigation events
        // For now, just increment event count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attosecond_laser::XRayImprint;
    use crate::k_parameter::KParameterState;

    #[tokio::test]
    async fn test_analytics_engine() {
        let mut engine = AnalyticsEngine::new();

        // Simulate some events
        let k_state = KParameterState::new(7.5);
        let imprint = XRayImprint {
            imprint_id: "test123".to_string(),
            hydrogen_bond_modulation: 0.8,
            lattice_distortion: [0.1, 0.0, 0.0],
            coherence_time_fs: 150.0,
            k_parameter_encoding: 7.5,
            tor_phase_signature: vec![1.0, 2.0, 3.0],
        };

        engine.record_thought_event(25.0, &k_state, &imprint).await;

        assert_eq!(engine.thought_count, 1);
        assert_eq!(engine.total_events(), 2); // Thought + laser events
    }

    #[test]
    fn test_cosmic_weather_types() {
        assert_eq!(WeatherType::QuantumCalm.emoji(), "☀️");
        assert_eq!(WeatherType::BraneStorm.emoji(), "⛈️");
        assert!(WeatherType::QuantumCalm.description().contains("Perfect"));
    }

    #[test]
    fn test_analytics_report_generation() {
        let engine = AnalyticsEngine::new();
        let report = engine.get_latest_report();

        assert!(!report.report_id.is_empty());
        assert_eq!(report.total_events, 0);
        assert!(report.uptime_seconds >= 0.0);
    }
}
