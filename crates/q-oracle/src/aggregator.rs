//! Quantum Price Aggregator
//!
//! Physics-inspired price aggregation with quantum mechanics principles

use crate::types::*;
use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Quantum Price Aggregator with physics-inspired algorithms
pub struct QuantumPriceAggregator {
    /// Configuration for quantum price aggregation
    config: Arc<RwLock<QuantumOracleConfig>>,
    /// Active price submissions for each feed
    submissions: Arc<RwLock<HashMap<String, Vec<QuantumPriceSubmission>>>>,
    /// Current aggregated prices
    aggregated_prices: Arc<RwLock<HashMap<String, QuantumAggregatedPrice>>>,
    /// Quantum state for each feed
    quantum_states: Arc<RwLock<HashMap<String, FeedQuantumState>>>,
    /// Performance metrics
    metrics: Arc<RwLock<AggregatorMetrics>>,
}

#[derive(Debug, Clone)]
struct QuantumPriceSubmission {
    oracle_id: String,
    value: BigDecimal,
    quantum_weight: f64,
    wave_amplitude: f64,
    timestamp: DateTime<Utc>,
    reputation_score: f64,
}

#[derive(Debug, Clone)]
struct QuantumAggregatedPrice {
    feed_id: String,
    price: BigDecimal,
    quantum_confidence: f64,
    wave_amplitude: f64,
    participants: u32,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct FeedQuantumState {
    superposition_active: bool,
    entangled_feeds: Vec<String>,
    coherence_level: f64,
    last_collapse: DateTime<Utc>,
    wave_evolution_rate: f64,
}

#[derive(Debug, Clone)]
struct AggregatorMetrics {
    total_submissions: u64,
    successful_aggregations: u64,
    average_confidence: f64,
    quantum_coherence_avg: f64,
    last_updated: DateTime<Utc>,
}

impl Default for AggregatorMetrics {
    fn default() -> Self {
        Self {
            total_submissions: 0,
            successful_aggregations: 0,
            average_confidence: 0.0,
            quantum_coherence_avg: 0.0,
            last_updated: Utc::now(),
        }
    }
}

impl QuantumPriceAggregator {
    /// Create new quantum price aggregator
    pub async fn new(config: &QuantumOracleConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config.clone())),
            submissions: Arc::new(RwLock::new(HashMap::new())),
            aggregated_prices: Arc::new(RwLock::new(HashMap::new())),
            quantum_states: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(AggregatorMetrics::default())),
        })
    }

    /// Initialize the aggregator
    pub async fn initialize(&self) -> Result<()> {
        info!("🔢 Initializing Quantum Price Aggregator");

        // Initialize quantum states for common feeds
        let common_feeds = vec!["ORB/USD", "ORBUSD/USD", "BTC/USD", "ETH/USD", "SOL/USD"];

        let mut quantum_states = self.quantum_states.write().await;

        for feed_id in common_feeds {
            let state = FeedQuantumState {
                superposition_active: true,
                entangled_feeds: Vec::new(),
                coherence_level: 1.0,
                last_collapse: Utc::now(),
                wave_evolution_rate: 1.0,
            };

            quantum_states.insert(feed_id.to_string(), state);
        }

        // Setup entanglement relationships
        self.setup_quantum_entanglements().await?;

        info!("✨ Quantum Price Aggregator initialized");
        Ok(())
    }

    /// Aggregate quantum price from submission
    pub async fn aggregate_quantum_price(
        &self,
        submission: &QuantumOracleSubmission,
        uncertainty_adjusted_value: BigDecimal,
    ) -> Result<QuantumAggregationResult> {
        debug!("⚖️  Aggregating quantum price for {}", submission.feed_id);

        // Create quantum price submission
        let quantum_submission = QuantumPriceSubmission {
            oracle_id: submission.oracle_id.clone(),
            value: uncertainty_adjusted_value.clone(),
            quantum_weight: self.calculate_quantum_weight(submission).await?,
            wave_amplitude: submission
                .wave_function_data
                .as_ref()
                .map(|w| w.amplitude)
                .unwrap_or(1.0),
            timestamp: submission.timestamp,
            reputation_score: submission.ai_confidence,
        };

        // Add to submissions
        {
            let mut submissions = self.submissions.write().await;
            let feed_submissions = submissions
                .entry(submission.feed_id.clone())
                .or_insert_with(Vec::new);
            feed_submissions.push(quantum_submission);

            // Keep only recent submissions (last 100)
            if feed_submissions.len() > 100 {
                feed_submissions.remove(0);
            }
        }

        // Perform quantum aggregation
        let aggregation_result = self
            .perform_quantum_aggregation(&submission.feed_id)
            .await?;

        // Update metrics
        self.update_aggregator_metrics().await?;

        Ok(aggregation_result)
    }

    /// Get current quantum price for feed
    pub async fn get_current_quantum_price(
        &self,
        feed_id: &str,
    ) -> Result<QuantumAggregationResult> {
        let aggregated_prices = self.aggregated_prices.read().await;

        if let Some(aggregated) = aggregated_prices.get(feed_id) {
            let quantum_states = self.quantum_states.read().await;
            let state = quantum_states.get(feed_id);

            Ok(QuantumAggregationResult {
                feed_id: feed_id.to_string(),
                quantum_price: aggregated.price.clone(),
                wave_amplitude: aggregated.wave_amplitude,
                wave_state: if state.map(|s| s.superposition_active).unwrap_or(false) {
                    QuantumState::Superposition
                } else {
                    QuantumState::Collapsed
                },
                ai_score: aggregated.quantum_confidence,
                uncertainty_bounds: self.calculate_uncertainty_bounds(&aggregated.price).await?,
                entangled_feeds: state.map(|s| s.entangled_feeds.clone()).unwrap_or_default(),
                final_price: aggregated.price.clone(),
                participants: vec![], // TODO: Track participants
                timestamp: aggregated.last_updated,
            })
        } else {
            Err(anyhow::anyhow!(format!(
                "No aggregated price found for feed: {}",
                feed_id
            )))
        }
    }

    /// Calculate Heisenberg uncertainty for all feeds
    pub async fn calculate_heisenberg_uncertainty(&self, uncertainty_factor: f64) -> Result<()> {
        debug!(
            "⚛️  Calculating Heisenberg uncertainty: factor={:.6}",
            uncertainty_factor
        );

        let mut quantum_states = self.quantum_states.write().await;

        for (feed_id, state) in quantum_states.iter_mut() {
            // Calculate position-momentum uncertainty: ΔxΔp ≥ ℏ/2
            let position_uncertainty = state.coherence_level * uncertainty_factor;
            let momentum_uncertainty = 6.62607015e-34 / (2.0 * position_uncertainty);

            // Update coherence based on uncertainty
            state.coherence_level = state.coherence_level * (1.0 - momentum_uncertainty.min(0.01));

            // Check if wave function should collapse
            if state.coherence_level < 0.1 {
                state.superposition_active = false;
                state.last_collapse = Utc::now();
                debug!(
                    "🌊 Wave function collapsed for {}: coherence={:.3}",
                    feed_id, state.coherence_level
                );
            }
        }

        Ok(())
    }

    /// Setup quantum entanglements between feeds
    async fn setup_quantum_entanglements(&self) -> Result<()> {
        info!("🔗 Setting up quantum feed entanglements");

        let entanglement_pairs = vec![("ORB/USD", "ORBUSD/USD"), ("BTC/USD", "ETH/USD")];

        let mut quantum_states = self.quantum_states.write().await;

        for (feed1, feed2) in entanglement_pairs {
            // Create bidirectional entanglement
            if let Some(state1) = quantum_states.get_mut(feed1) {
                if !state1.entangled_feeds.contains(&feed2.to_string()) {
                    state1.entangled_feeds.push(feed2.to_string());
                }
            }

            if let Some(state2) = quantum_states.get_mut(feed2) {
                if !state2.entangled_feeds.contains(&feed1.to_string()) {
                    state2.entangled_feeds.push(feed1.to_string());
                }
            }
        }

        Ok(())
    }

    /// Calculate quantum weight for submission
    async fn calculate_quantum_weight(&self, submission: &QuantumOracleSubmission) -> Result<f64> {
        // Base weight from oracle reputation
        let mut weight = submission.ai_confidence;

        // Quantum enhancement based on wave function data
        if let Some(ref wave_data) = submission.wave_function_data {
            // Higher amplitude = higher confidence = higher weight
            weight *= wave_data.amplitude;

            // Coherent states get bonus weight
            if matches!(
                wave_data.quantum_state,
                QuantumState::Superposition | QuantumState::Entangled
            ) {
                weight *= 1.1;
            }
        }

        // Time-based decay (recent submissions weighted higher)
        let age_seconds = (Utc::now() - submission.timestamp).num_seconds() as f64;
        let time_decay = (-age_seconds / 300.0).exp(); // 5-minute half-life
        weight *= time_decay;

        Ok(weight.max(0.01).min(2.0)) // Bounded between 0.01 and 2.0
    }

    /// Perform quantum aggregation for a feed
    async fn perform_quantum_aggregation(&self, feed_id: &str) -> Result<QuantumAggregationResult> {
        let submissions = self.submissions.read().await;
        let feed_submissions = submissions.get(feed_id);

        if feed_submissions.is_none() || feed_submissions.unwrap().is_empty() {
            return Err(anyhow::anyhow!(format!(
                "No submissions found for feed: {}",
                feed_id
            )));
        }

        let submissions_list = feed_submissions.unwrap();

        // Quantum weighted average with superposition
        let mut total_weight = 0.0;
        let mut weighted_sum = BigDecimal::from(0);
        let mut wave_amplitude_sum = 0.0;

        for submission in submissions_list {
            let weight = submission.quantum_weight;
            total_weight += weight;
            let weight_decimal = BigDecimal::from_str(&weight.to_string()).map_err(|e| anyhow::anyhow!("Failed to convert weight: {}", e))?;
            weighted_sum += &submission.value * weight_decimal;
            wave_amplitude_sum += submission.wave_amplitude * weight;
        }

        let quantum_price = if total_weight > 0.0 {
            let total_weight_decimal = BigDecimal::from_str(&total_weight.to_string()).map_err(|e| anyhow::anyhow!("Failed to convert total_weight: {}", e))?;
            weighted_sum / total_weight_decimal
        } else {
            BigDecimal::from(0)
        };

        let avg_wave_amplitude = if total_weight > 0.0 {
            wave_amplitude_sum / total_weight
        } else {
            0.0
        };

        // Determine quantum state
        let quantum_states = self.quantum_states.read().await;
        let state = quantum_states.get(feed_id);
        let wave_state = if let Some(state) = state {
            if state.superposition_active {
                QuantumState::Superposition
            } else {
                QuantumState::Collapsed
            }
        } else {
            QuantumState::Collapsed
        };

        // Calculate AI confidence score
        let ai_score = submissions_list
            .iter()
            .map(|s| s.reputation_score * s.quantum_weight)
            .sum::<f64>()
            / total_weight.max(1.0);

        // Update aggregated price
        {
            let mut aggregated_prices = self.aggregated_prices.write().await;
            aggregated_prices.insert(
                feed_id.to_string(),
                QuantumAggregatedPrice {
                    feed_id: feed_id.to_string(),
                    price: quantum_price.clone(),
                    quantum_confidence: ai_score,
                    wave_amplitude: avg_wave_amplitude,
                    participants: submissions_list.len() as u32,
                    last_updated: Utc::now(),
                },
            );
        }

        Ok(QuantumAggregationResult {
            feed_id: feed_id.to_string(),
            quantum_price: quantum_price.clone(),
            wave_amplitude: avg_wave_amplitude,
            wave_state,
            ai_score,
            uncertainty_bounds: self.calculate_uncertainty_bounds(&quantum_price).await?,
            entangled_feeds: state.map(|s| s.entangled_feeds.clone()).unwrap_or_default(),
            final_price: quantum_price,
            participants: submissions_list
                .iter()
                .map(|s| s.oracle_id.clone())
                .collect(),
            timestamp: Utc::now(),
        })
    }

    /// Calculate uncertainty bounds using quantum principles
    async fn calculate_uncertainty_bounds(
        &self,
        price: &BigDecimal,
    ) -> Result<(BigDecimal, BigDecimal)> {
        let config = self.config.read().await;
        let uncertainty_factor = BigDecimal::from_str(&config.uncertainty_factor.to_string())
            .map_err(|e| anyhow::anyhow!("Failed to convert uncertainty_factor: {}", e))?;

        let uncertainty = price * &uncertainty_factor;
        let lower_bound = price - &uncertainty;
        let upper_bound = price + uncertainty;

        Ok((lower_bound, upper_bound))
    }

    /// Update aggregator metrics
    async fn update_aggregator_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        metrics.total_submissions += 1;

        // Calculate average confidence from all feeds
        let aggregated_prices = self.aggregated_prices.read().await;
        if !aggregated_prices.is_empty() {
            let total_confidence: f64 = aggregated_prices
                .values()
                .map(|p| p.quantum_confidence)
                .sum();
            metrics.average_confidence = total_confidence / aggregated_prices.len() as f64;
            metrics.successful_aggregations += 1;
        }

        // Calculate average quantum coherence
        let quantum_states = self.quantum_states.read().await;
        if !quantum_states.is_empty() {
            let total_coherence: f64 = quantum_states.values().map(|s| s.coherence_level).sum();
            metrics.quantum_coherence_avg = total_coherence / quantum_states.len() as f64;
        }

        metrics.last_updated = Utc::now();
        Ok(())
    }
}
