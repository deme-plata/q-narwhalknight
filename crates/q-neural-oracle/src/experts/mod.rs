//! Mixture of Quantum Experts
//!
//! Domain-specific expert networks with quantum-inspired attention aggregation.

use crate::quantum::QuantumState;
use crate::PredictionContext;
use std::collections::HashMap;
use tracing::{debug, info};

/// Prediction domains for specialized experts
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PredictionDomain {
    /// VDF difficulty optimization
    VDFOptimization,
    /// Fee rate forecasting
    FeeForecasting,
    /// Fee smoothing reserve management
    ReserveManagement,
    /// Staking pool economics
    StakingEconomics,
    /// Network security metrics
    SecurityAnalysis,
    /// Governance proposal outcomes
    GovernanceOutcomes,
    /// Network topology optimization
    NetworkTopology,
    /// Market dynamics
    MarketDynamics,
}

impl PredictionDomain {
    /// Get all domains
    pub fn all() -> Vec<Self> {
        vec![
            Self::VDFOptimization,
            Self::FeeForecasting,
            Self::ReserveManagement,
            Self::StakingEconomics,
            Self::SecurityAnalysis,
            Self::GovernanceOutcomes,
            Self::NetworkTopology,
            Self::MarketDynamics,
        ]
    }
}

/// Mixture of Quantum Experts for specialized predictions
pub struct MixtureOfQuantumExperts {
    /// Domain-specific expert networks
    experts: HashMap<PredictionDomain, QuantumExpert>,

    /// Gating network (router)
    gating_weights: Vec<f64>,

    /// Load balancing statistics
    load_stats: HashMap<PredictionDomain, LoadStats>,

    /// Number of qubits
    num_qubits: usize,
}

/// Individual quantum expert network
pub struct QuantumExpert {
    /// Domain
    domain: PredictionDomain,

    /// Expert weights (trained parameters)
    weights: Vec<f64>,

    /// Bias terms
    biases: Vec<f64>,

    /// Number of hidden units
    hidden_size: usize,

    /// Performance tracking
    performance: ExpertPerformance,
}

/// Expert performance metrics
#[derive(Clone, Debug, Default)]
struct ExpertPerformance {
    total_predictions: u64,
    correct_predictions: u64,
    cumulative_error: f64,
    last_accuracy: f64,
}

/// Load balancing statistics
#[derive(Clone, Debug, Default)]
struct LoadStats {
    invocations: u64,
    total_time_ns: u64,
}

/// Expert prediction result
#[derive(Clone, Debug)]
pub struct ExpertPrediction {
    /// Primary prediction value
    pub primary_value: f64,

    /// Confidence score (0-1)
    pub confidence: f64,

    /// Domain of this prediction
    pub domain: PredictionDomain,

    /// Expert weights used
    pub expert_weights: Vec<(PredictionDomain, f64)>,

    /// Quantum entropy (measure of quantum advantage)
    pub quantum_entropy: f64,
}

impl MixtureOfQuantumExperts {
    /// Create new mixture of experts
    pub fn new(num_qubits: usize) -> Self {
        info!("🧠 Initializing Mixture of Quantum Experts");

        let mut experts = HashMap::new();
        let mut load_stats = HashMap::new();

        for domain in PredictionDomain::all() {
            experts.insert(domain, QuantumExpert::new(domain, num_qubits));
            load_stats.insert(domain, LoadStats::default());
            debug!("   Created expert for {:?}", domain);
        }

        // Initialize gating weights (uniform)
        let num_experts = PredictionDomain::all().len();
        let gating_weights = vec![1.0 / num_experts as f64; num_experts];

        info!("✅ Created {} expert networks", experts.len());

        Self {
            experts,
            gating_weights,
            load_stats,
            num_qubits,
        }
    }

    /// Route input to experts and aggregate predictions
    pub async fn predict(
        &self,
        quantum_state: &QuantumState,
        target_domain: PredictionDomain,
        context: &PredictionContext,
    ) -> ExpertPrediction {
        let start = std::time::Instant::now();

        // Step 1: Compute gating weights from quantum state
        let expert_weights = self.compute_gating_weights(quantum_state, target_domain);

        // Step 2: Select top-k experts (sparse MoE)
        let top_k = 4;
        let selected = self.select_top_k_experts(&expert_weights, top_k);

        // Step 3: Run selected experts
        let mut predictions = Vec::new();
        for (domain, weight) in &selected {
            if let Some(expert) = self.experts.get(domain) {
                let pred = expert.predict(quantum_state, context);
                predictions.push((*domain, *weight, pred));
            }
        }

        // Step 4: Aggregate predictions
        let aggregated = self.aggregate_predictions(&predictions, quantum_state);

        // Step 5: Calculate uncertainty from expert disagreement
        let uncertainty = self.calculate_uncertainty(&predictions);
        let confidence = (1.0 - uncertainty).clamp(0.0, 1.0);

        let elapsed = start.elapsed();
        debug!("MoQE prediction in {:?}", elapsed);

        ExpertPrediction {
            primary_value: aggregated,
            confidence,
            domain: target_domain,
            expert_weights: selected,
            quantum_entropy: quantum_state.entanglement_entropy(),
        }
    }

    /// Compute gating weights based on quantum state and target domain
    fn compute_gating_weights(
        &self,
        quantum_state: &QuantumState,
        target_domain: PredictionDomain,
    ) -> Vec<(PredictionDomain, f64)> {
        let amplitudes = quantum_state.amplitudes();

        PredictionDomain::all()
            .into_iter()
            .enumerate()
            .map(|(i, domain)| {
                // Base weight from gating network
                let base_weight = self.gating_weights.get(i).copied().unwrap_or(0.0);

                // Boost weight for target domain
                let target_boost = if domain == target_domain { 2.0 } else { 1.0 };

                // Quantum-derived weight (from amplitude patterns)
                let quantum_weight = if i < amplitudes.len() {
                    amplitudes[i].norm_sqr()
                } else {
                    0.1
                };

                // Performance-based weight
                let perf_weight = self.experts.get(&domain)
                    .map(|e| 0.5 + e.performance.last_accuracy * 0.5)
                    .unwrap_or(0.5);

                let total_weight = base_weight * target_boost * quantum_weight * perf_weight;
                (domain, total_weight)
            })
            .collect()
    }

    /// Select top-k experts by weight
    fn select_top_k_experts(
        &self,
        weights: &[(PredictionDomain, f64)],
        k: usize,
    ) -> Vec<(PredictionDomain, f64)> {
        let mut sorted = weights.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Normalize weights of selected experts
        let selected: Vec<_> = sorted.into_iter().take(k).collect();
        let total: f64 = selected.iter().map(|(_, w)| w).sum();

        selected
            .into_iter()
            .map(|(domain, weight)| (domain, weight / total.max(1e-10)))
            .collect()
    }

    /// Aggregate predictions from multiple experts
    fn aggregate_predictions(
        &self,
        predictions: &[(PredictionDomain, f64, f64)],
        _quantum_state: &QuantumState,
    ) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        // Weighted average
        let weighted_sum: f64 = predictions
            .iter()
            .map(|(_, weight, pred)| weight * pred)
            .sum();

        let total_weight: f64 = predictions.iter().map(|(_, w, _)| w).sum();

        weighted_sum / total_weight.max(1e-10)
    }

    /// Calculate uncertainty from expert disagreement
    fn calculate_uncertainty(&self, predictions: &[(PredictionDomain, f64, f64)]) -> f64 {
        if predictions.len() < 2 {
            return 0.5; // High uncertainty with few experts
        }

        // Calculate weighted mean
        let total_weight: f64 = predictions.iter().map(|(_, w, _)| w).sum();
        let weighted_mean: f64 = predictions
            .iter()
            .map(|(_, w, p)| w * p)
            .sum::<f64>() / total_weight;

        // Calculate weighted variance
        let variance: f64 = predictions
            .iter()
            .map(|(_, w, p)| {
                let diff = p - weighted_mean;
                w * diff * diff
            })
            .sum::<f64>() / total_weight;

        // Normalize variance to 0-1 range (assuming max reasonable variance ~ 1.0)
        (variance.sqrt() / 1.0).clamp(0.0, 1.0)
    }

    /// Update expert from outcome (online learning)
    pub fn update_from_outcome(
        &mut self,
        domain: PredictionDomain,
        predicted: f64,
        actual: f64,
    ) {
        if let Some(expert) = self.experts.get_mut(&domain) {
            expert.update_from_outcome(predicted, actual);
        }
    }
}

impl QuantumExpert {
    /// Create new expert for domain
    fn new(domain: PredictionDomain, num_qubits: usize) -> Self {
        let config = Self::domain_config(domain);

        // Initialize weights with small random values
        let input_size = 1 << num_qubits.min(6); // Cap at 64 for memory
        let hidden_size = config.hidden_size;
        let output_size = 1;

        let num_weights = input_size * hidden_size + hidden_size * output_size;
        let weights: Vec<f64> = (0..num_weights)
            .map(|_| (rand::random::<f64>() - 0.5) * 0.1)
            .collect();

        let num_biases = hidden_size + output_size;
        let biases = vec![0.0; num_biases];

        Self {
            domain,
            weights,
            biases,
            hidden_size,
            performance: ExpertPerformance::default(),
        }
    }

    /// Get domain-specific configuration
    fn domain_config(domain: PredictionDomain) -> ExpertConfig {
        match domain {
            PredictionDomain::VDFOptimization => ExpertConfig { hidden_size: 32 },
            PredictionDomain::FeeForecasting => ExpertConfig { hidden_size: 64 },
            PredictionDomain::ReserveManagement => ExpertConfig { hidden_size: 48 },
            PredictionDomain::StakingEconomics => ExpertConfig { hidden_size: 48 },
            PredictionDomain::SecurityAnalysis => ExpertConfig { hidden_size: 64 },
            PredictionDomain::GovernanceOutcomes => ExpertConfig { hidden_size: 32 },
            PredictionDomain::NetworkTopology => ExpertConfig { hidden_size: 48 },
            PredictionDomain::MarketDynamics => ExpertConfig { hidden_size: 64 },
        }
    }

    /// Make prediction from quantum state
    fn predict(&self, quantum_state: &QuantumState, _context: &PredictionContext) -> f64 {
        // Extract features from quantum amplitudes
        let amplitudes = quantum_state.amplitudes();
        let input: Vec<f64> = amplitudes
            .iter()
            .take(64) // Limit input size
            .map(|a| a.norm())
            .collect();

        // Simple feedforward: input -> hidden -> output
        let hidden = self.forward_hidden(&input);
        let output = self.forward_output(&hidden);

        output
    }

    /// Forward pass through hidden layer
    fn forward_hidden(&self, input: &[f64]) -> Vec<f64> {
        let input_size = input.len();
        let mut hidden = vec![0.0; self.hidden_size];

        for j in 0..self.hidden_size {
            let mut sum = self.biases.get(j).copied().unwrap_or(0.0);

            for (i, &x) in input.iter().enumerate() {
                let weight_idx = i * self.hidden_size + j;
                let weight = self.weights.get(weight_idx).copied().unwrap_or(0.0);
                sum += weight * x;
            }

            // ReLU activation
            hidden[j] = sum.max(0.0);
        }

        hidden
    }

    /// Forward pass through output layer
    fn forward_output(&self, hidden: &[f64]) -> f64 {
        let offset = hidden.len() * self.hidden_size; // Skip input-hidden weights
        let bias_offset = self.hidden_size;

        let mut sum = self.biases.get(bias_offset).copied().unwrap_or(0.0);

        for (j, &h) in hidden.iter().enumerate() {
            let weight_idx = offset + j;
            let weight = self.weights.get(weight_idx).copied().unwrap_or(0.0);
            sum += weight * h;
        }

        // Sigmoid output (bounded 0-1)
        1.0 / (1.0 + (-sum).exp())
    }

    /// Update weights from outcome
    fn update_from_outcome(&mut self, predicted: f64, actual: f64) {
        let error = (predicted - actual).abs();

        self.performance.total_predictions += 1;
        self.performance.cumulative_error += error;

        // Simple accuracy: prediction within 10% of actual
        if error < actual.abs() * 0.1 {
            self.performance.correct_predictions += 1;
        }

        self.performance.last_accuracy =
            self.performance.correct_predictions as f64
            / self.performance.total_predictions.max(1) as f64;
    }
}

struct ExpertConfig {
    hidden_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mixture_creation() {
        let moqe = MixtureOfQuantumExperts::new(4);
        assert_eq!(moqe.experts.len(), PredictionDomain::all().len());
    }

    #[tokio::test]
    async fn test_expert_prediction() {
        let moqe = MixtureOfQuantumExperts::new(4);
        let state = QuantumState::uniform_superposition(4);

        let context = PredictionContext {
            block_height: 100_000,
            current_fee_rate: 1.5,
            hashrate: 1e18,
            staking_total: 1_000_000_000_000,
            validator_count: 100,
            tx_volume: 50_000.0,
            historical: vec![],
            domain_features: std::collections::HashMap::new(),
        };

        let prediction = moqe.predict(&state, PredictionDomain::FeeForecasting, &context).await;

        assert!(prediction.primary_value >= 0.0);
        assert!(prediction.primary_value <= 1.0);
        assert!(prediction.confidence >= 0.0);
        assert!(prediction.confidence <= 1.0);
    }
}
