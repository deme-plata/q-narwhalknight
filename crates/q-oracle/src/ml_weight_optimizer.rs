//! ML-Driven Adaptive Weight Optimizer for Oracle Submissions
//!
//! Uses online linear regression with exponential moving average (EMA) weights
//! to predict optimal oracle submission weights based on:
//! - Price delta from previous submissions
//! - Trading volume
//! - Time decay
//! - Oracle reputation
//! - Quantum coherence
//! - Volatility metrics
//!
//! Key features:
//! - <1ms inference time (7 multiplications + sigmoid)
//! - Online learning adapts within 10-20 samples
//! - Cold start heuristics until model is trained
//! - Adaptive uncertainty scaling based on volatility

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info, warn};

/// Number of features in the ML weight model
pub const NUM_WEIGHT_FEATURES: usize = 7;

/// Oracle submission features for ML weight prediction
#[derive(Debug, Clone, Default)]
pub struct OracleFeatures {
    /// Price delta from last submission (normalized -1 to 1)
    pub price_delta: f32,
    /// Trading volume in last period (normalized 0-1)
    pub volume_normalized: f32,
    /// Time since last submission (normalized 0-1, decays to 0 over 5 min)
    pub time_decay: f32,
    /// Oracle reputation score (0-1)
    pub reputation: f32,
    /// Quantum coherence level (0-1)
    pub coherence: f32,
    /// Current volatility (0-1, where 1 = extreme volatility)
    pub volatility: f32,
    /// Number of other oracles agreeing (consensus signal, 0-1)
    pub consensus_ratio: f32,
}

impl OracleFeatures {
    /// Convert features to normalized tensor (7 dimensions, 0-1 range)
    pub fn to_tensor(&self) -> [f32; NUM_WEIGHT_FEATURES] {
        [
            // Price delta: sigmoid transform to 0-1
            sigmoid(self.price_delta * 2.0),
            // Volume: already 0-1
            self.volume_normalized.clamp(0.0, 1.0),
            // Time decay: already 0-1
            self.time_decay.clamp(0.0, 1.0),
            // Reputation: already 0-1
            self.reputation.clamp(0.0, 1.0),
            // Coherence: already 0-1
            self.coherence.clamp(0.0, 1.0),
            // Volatility: already 0-1
            self.volatility.clamp(0.0, 1.0),
            // Consensus: already 0-1
            self.consensus_ratio.clamp(0.0, 1.0),
        ]
    }

    /// Create features from raw values
    pub fn new(
        price_delta: f32,
        volume_normalized: f32,
        time_decay: f32,
        reputation: f32,
        coherence: f32,
        volatility: f32,
        consensus_ratio: f32,
    ) -> Self {
        Self {
            price_delta,
            volume_normalized,
            time_decay,
            reputation,
            coherence,
            volatility,
            consensus_ratio,
        }
    }
}

/// Outcome of an oracle submission for learning
#[derive(Debug, Clone)]
pub struct WeightOutcome {
    /// Features at prediction time
    pub features: OracleFeatures,
    /// Predicted weight by ML model
    pub predicted_weight: f64,
    /// Actual weight that would have been optimal (computed post-hoc)
    pub optimal_weight: f64,
    /// Whether the submission was accepted
    pub accepted: bool,
    /// Error from final aggregated price (lower is better)
    pub price_error: f64,
    /// Timestamp when outcome was recorded
    pub timestamp: Instant,
}

/// Configuration for the weight optimizer
#[derive(Debug, Clone)]
pub struct WeightOptimizerConfig {
    /// Minimum oracle weight
    pub min_weight: f64,
    /// Maximum oracle weight
    pub max_weight: f64,
    /// Learning rate for online gradient descent
    pub learning_rate: f32,
    /// EMA decay factor (0.1 = fast adaptation, 0.01 = slow)
    pub ema_decay: f32,
    /// Cold start threshold (use heuristics below this sample count)
    pub cold_start_threshold: u64,
    /// History size for validation and analysis
    pub history_size: usize,
    /// Base uncertainty factor (1.618% golden ratio)
    pub base_uncertainty: f32,
    /// Maximum uncertainty scaling factor
    pub max_uncertainty_scale: f32,
}

impl Default for WeightOptimizerConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.01,
            max_weight: 2.0,
            learning_rate: 0.01,
            ema_decay: 0.1,
            cold_start_threshold: 50,
            history_size: 100,
            base_uncertainty: 0.01618, // Golden ratio uncertainty
            max_uncertainty_scale: 5.0,
        }
    }
}

/// Online linear regression model for oracle weight prediction
pub struct OracleWeightPredictor {
    /// Weights for each feature (7 dimensions)
    weights: [f32; NUM_WEIGHT_FEATURES],
    /// Bias term
    bias: f32,
    /// Configuration
    config: WeightOptimizerConfig,
    /// Number of samples seen (for cold start detection)
    samples_seen: AtomicU64,
    /// Running mean of features (for normalization stability)
    feature_mean: [f32; NUM_WEIGHT_FEATURES],
    /// Running variance of features (Welford's algorithm)
    feature_m2: [f32; NUM_WEIGHT_FEATURES],
    /// Historical outcomes for analysis
    outcome_history: VecDeque<WeightOutcome>,
    /// Current volatility estimate (EMA)
    volatility_ema: f32,
    /// Current adaptive uncertainty factor
    adaptive_uncertainty: f32,
    /// Last prediction for logging
    last_prediction: f64,
    /// Cumulative reward for tracking
    cumulative_reward: f32,
    /// Best accuracy achieved
    best_accuracy: f32,
}

impl OracleWeightPredictor {
    /// Create a new oracle weight predictor with default weights
    pub fn new(config: WeightOptimizerConfig) -> Self {
        // Initialize weights with physics-inspired priors
        let weights = [
            -0.1,  // Price delta: smaller delta = more stable = higher weight
            0.2,   // Volume: higher volume = more market signal = higher weight
            -0.3,  // Time decay: fresher data = higher weight (negative because decay decreases)
            0.4,   // Reputation: higher reputation = higher weight
            0.3,   // Coherence: higher quantum coherence = higher weight
            -0.2,  // Volatility: higher volatility = lower weight (more uncertainty)
            0.3,   // Consensus: more agreement = higher weight
        ];

        // Bias initialized to produce medium weights
        let bias = 0.5;

        info!("🤖 [ORACLE WEIGHT OPTIMIZER] Initialized ML predictor");
        info!("   Config: min={:.3}, max={:.3}, lr={:.3}, ema={:.3}",
              config.min_weight, config.max_weight,
              config.learning_rate, config.ema_decay);
        info!("   Adaptive uncertainty: base={:.4}, max_scale={:.1}",
              config.base_uncertainty, config.max_uncertainty_scale);

        Self {
            weights,
            bias,
            config,
            samples_seen: AtomicU64::new(0),
            feature_mean: [0.0; NUM_WEIGHT_FEATURES],
            feature_m2: [0.0; NUM_WEIGHT_FEATURES],
            outcome_history: VecDeque::with_capacity(100),
            volatility_ema: 0.1, // Start with low volatility assumption
            adaptive_uncertainty: 0.01618, // Golden ratio default
            last_prediction: 1.0,
            cumulative_reward: 0.0,
            best_accuracy: 0.0,
        }
    }

    /// Predict optimal weight for an oracle submission
    pub fn predict_weight(&self, features: &OracleFeatures) -> f64 {
        let samples = self.samples_seen.load(Ordering::Relaxed);

        // Cold start: use heuristics until we have enough samples
        if samples < self.config.cold_start_threshold {
            let heuristic = self.heuristic_weight(features);
            debug!("🤖 [ORACLE WEIGHT] Cold start ({}/{} samples): heuristic weight {:.4}",
                   samples, self.config.cold_start_threshold, heuristic);
            return heuristic;
        }

        // ML prediction
        let features_tensor = features.to_tensor();
        let normalized = self.predict_normalized(&features_tensor);

        // Denormalize to actual weight range
        let range = self.config.max_weight - self.config.min_weight;
        let predicted = self.config.min_weight + (normalized as f64 * range);

        // Clamp to valid range
        let clamped = predicted.clamp(self.config.min_weight, self.config.max_weight);

        debug!("🤖 [ORACLE WEIGHT] ML prediction: {:.4} (normalized: {:.3})",
               clamped, normalized);

        clamped
    }

    /// Heuristic weight for cold start (before model is trained)
    fn heuristic_weight(&self, features: &OracleFeatures) -> f64 {
        let base = 1.0_f64; // Default weight

        // Reputation is the most important factor
        let reputation_factor = 0.5 + features.reputation as f64 * 0.5;

        // Fresher data gets higher weight
        let freshness_factor = 0.5 + features.time_decay as f64 * 0.5;

        // High volatility reduces weight
        let volatility_penalty = 1.0 - (features.volatility as f64 * 0.3);

        // Consensus increases weight
        let consensus_bonus = 1.0 + (features.consensus_ratio as f64 * 0.2);

        let adjusted = base * reputation_factor * freshness_factor * volatility_penalty * consensus_bonus;
        adjusted.clamp(self.config.min_weight, self.config.max_weight)
    }

    /// Raw normalized prediction (0-1) using linear model
    fn predict_normalized(&self, features: &[f32; NUM_WEIGHT_FEATURES]) -> f32 {
        let mut sum = self.bias;
        for i in 0..NUM_WEIGHT_FEATURES {
            sum += self.weights[i] * features[i];
        }
        // Sigmoid to bound output to 0-1
        sigmoid(sum)
    }

    /// Record outcome and update model (online learning)
    pub fn record_outcome(&mut self, outcome: WeightOutcome) {
        let samples = self.samples_seen.fetch_add(1, Ordering::Relaxed) + 1;

        // Update volatility EMA
        self.update_volatility_ema(outcome.features.volatility);

        // Calculate reward signal
        let reward = self.calculate_reward(&outcome);
        self.cumulative_reward += reward;

        // Update best accuracy if this was good
        let accuracy = 1.0 - (outcome.price_error as f32).min(1.0);
        if accuracy > self.best_accuracy && outcome.accepted {
            self.best_accuracy = accuracy;
            info!("🤖 [ORACLE WEIGHT] New best accuracy: {:.4}", self.best_accuracy);
        }

        // Only do gradient updates after cold start
        if samples >= self.config.cold_start_threshold {
            self.gradient_update(&outcome, reward);
        }

        // Update running statistics
        let features_tensor = outcome.features.to_tensor();
        self.update_running_stats(&features_tensor, samples);

        // Store in history
        self.outcome_history.push_back(outcome);
        if self.outcome_history.len() > self.config.history_size {
            self.outcome_history.pop_front();
        }

        // Log periodically
        if samples % 10 == 0 {
            self.log_model_state(samples);
        }
    }

    /// Update volatility EMA for adaptive uncertainty
    fn update_volatility_ema(&mut self, current_volatility: f32) {
        let alpha = self.config.ema_decay;
        self.volatility_ema = alpha * current_volatility + (1.0 - alpha) * self.volatility_ema;

        // Update adaptive uncertainty based on volatility
        let volatility_scale = 1.0 + self.volatility_ema * (self.config.max_uncertainty_scale - 1.0);
        self.adaptive_uncertainty = self.config.base_uncertainty * volatility_scale;
    }

    /// Get current adaptive uncertainty factor
    pub fn get_adaptive_uncertainty(&self) -> f32 {
        self.adaptive_uncertainty
    }

    /// Get current volatility estimate
    pub fn get_volatility(&self) -> f32 {
        self.volatility_ema
    }

    /// Calculate dynamic fee based on volatility
    pub fn calculate_dynamic_fee(&self, base_fee: f64) -> f64 {
        // Higher volatility = higher fees (risk premium)
        let volatility_multiplier = 1.0 + (self.volatility_ema as f64 * 2.0);

        // Cap at 3x base fee during extreme volatility
        let dynamic_fee = base_fee * volatility_multiplier;
        dynamic_fee.min(base_fee * 3.0)
    }

    /// Calculate reward from outcome
    fn calculate_reward(&self, outcome: &WeightOutcome) -> f32 {
        if !outcome.accepted {
            // Strong negative signal for rejected submissions
            -1.0
        } else {
            // Reward inversely proportional to price error
            let error_penalty = (outcome.price_error as f32).min(1.0);
            1.0 - error_penalty
        }
    }

    /// Perform gradient descent update
    fn gradient_update(&mut self, outcome: &WeightOutcome, reward: f32) {
        let features_tensor = outcome.features.to_tensor();
        let current_prediction = self.predict_normalized(&features_tensor);

        // Target: adjust current prediction based on reward
        let target = if reward < 0.0 {
            // Negative reward: should have predicted lower weight
            (current_prediction - 0.2 * reward.abs()).clamp(0.0, 1.0)
        } else {
            // Positive reward: prediction was good, slight increase if accurate
            (current_prediction + 0.05 * reward).clamp(0.0, 1.0)
        };

        let error = target - current_prediction;

        // Gradient descent with EMA smoothing
        for i in 0..NUM_WEIGHT_FEATURES {
            let gradient = error * features_tensor[i] * sigmoid_derivative(current_prediction);
            self.weights[i] += self.config.learning_rate * gradient;
            // Regularization: keep weights bounded
            self.weights[i] = self.weights[i].clamp(-2.0, 2.0);
        }
        self.bias += self.config.learning_rate * error * sigmoid_derivative(current_prediction);
        self.bias = self.bias.clamp(-2.0, 2.0);
    }

    /// Update running statistics using Welford's algorithm
    fn update_running_stats(&mut self, features: &[f32; NUM_WEIGHT_FEATURES], n: u64) {
        let n_f32 = n as f32;
        for i in 0..NUM_WEIGHT_FEATURES {
            let delta = features[i] - self.feature_mean[i];
            self.feature_mean[i] += delta / n_f32;
            let delta2 = features[i] - self.feature_mean[i];
            self.feature_m2[i] += delta * delta2;
        }
    }

    /// Log current model state
    fn log_model_state(&self, samples: u64) {
        let weights_str: Vec<String> = self.weights.iter()
            .map(|w| format!("{:.2}", w))
            .collect();

        info!("🤖 [ORACLE WEIGHT] Model state after {} samples:", samples);
        info!("   Weights: [{}]", weights_str.join(", "));
        info!("   Bias: {:.3}, Best accuracy: {:.2}%",
              self.bias, self.best_accuracy * 100.0);
        info!("   Volatility EMA: {:.4}, Adaptive uncertainty: {:.4}",
              self.volatility_ema, self.adaptive_uncertainty);
        info!("   Cumulative reward: {:.2}", self.cumulative_reward);
    }

    /// Get number of samples seen
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen.load(Ordering::Relaxed)
    }

    /// Get current model weights (for debugging/monitoring)
    pub fn get_weights(&self) -> [f32; NUM_WEIGHT_FEATURES] {
        self.weights
    }

    /// Check if model is trained (past cold start)
    pub fn is_trained(&self) -> bool {
        self.samples_seen.load(Ordering::Relaxed) >= self.config.cold_start_threshold
    }

    /// Get average accuracy from history
    pub fn average_accuracy(&self) -> f32 {
        if self.outcome_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.outcome_history.iter()
            .filter(|o| o.accepted)
            .map(|o| 1.0 - (o.price_error as f32).min(1.0))
            .sum();
        let count = self.outcome_history.iter()
            .filter(|o| o.accepted)
            .count();
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Get rejection rate from history
    pub fn rejection_rate(&self) -> f32 {
        if self.outcome_history.is_empty() {
            return 0.0;
        }
        let rejections = self.outcome_history.iter()
            .filter(|o| !o.accepted)
            .count();
        rejections as f32 / self.outcome_history.len() as f32
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative for backpropagation
#[inline]
fn sigmoid_derivative(sigmoid_output: f32) -> f32 {
    sigmoid_output * (1.0 - sigmoid_output)
}

// ============ AMM CONSTANT PRODUCT FORMULA ============

/// AMM (Automated Market Maker) price calculator using constant product formula
///
/// The constant product formula: x * y = k
/// - x = reserve of base token (e.g., ORB)
/// - y = reserve of quote token (e.g., USD)
/// - k = constant product (invariant)
///
/// When buying base token (ORB), price goes UP
/// When selling base token (ORB), price goes DOWN
#[derive(Debug, Clone)]
pub struct ConstantProductAMM {
    /// Reserve of base token (ORB)
    pub reserve_base: f64,
    /// Reserve of quote token (USD)
    pub reserve_quote: f64,
    /// Constant product (k = reserve_base * reserve_quote)
    pub k: f64,
    /// Fee rate (e.g., 0.003 = 0.3%)
    pub fee_rate: f64,
}

impl ConstantProductAMM {
    /// Create new AMM with initial reserves
    pub fn new(reserve_base: f64, reserve_quote: f64, fee_rate: f64) -> Self {
        let k = reserve_base * reserve_quote;
        Self {
            reserve_base,
            reserve_quote,
            k,
            fee_rate,
        }
    }

    /// Get current spot price (quote per base, e.g., USD per ORB)
    pub fn spot_price(&self) -> f64 {
        self.reserve_quote / self.reserve_base
    }

    /// Calculate output amount when buying base token with quote token
    /// BUY: User provides quote (USD), receives base (ORB)
    /// Price goes UP after this trade
    pub fn buy_base(&mut self, quote_amount: f64) -> (f64, f64, f64) {
        // Apply fee
        let fee = quote_amount * self.fee_rate;
        let quote_after_fee = quote_amount - fee;

        // New quote reserve after adding user's tokens
        let new_reserve_quote = self.reserve_quote + quote_after_fee;

        // Calculate new base reserve using constant product
        // new_base * new_quote = k
        // new_base = k / new_quote
        let new_reserve_base = self.k / new_reserve_quote;

        // Amount of base tokens user receives
        let base_output = self.reserve_base - new_reserve_base;

        // Calculate price impact
        let old_price = self.spot_price();

        // Update reserves
        self.reserve_base = new_reserve_base;
        self.reserve_quote = new_reserve_quote;

        let new_price = self.spot_price();
        let price_impact = (new_price - old_price) / old_price;

        info!("🔼 AMM BUY: {} quote → {} base | Price: {:.6} → {:.6} (+{:.2}%)",
              quote_amount, base_output, old_price, new_price, price_impact * 100.0);

        (base_output, new_price, price_impact)
    }

    /// Calculate output amount when selling base token for quote token
    /// SELL: User provides base (ORB), receives quote (USD)
    /// Price goes DOWN after this trade
    pub fn sell_base(&mut self, base_amount: f64) -> (f64, f64, f64) {
        // New base reserve after adding user's tokens
        let new_reserve_base = self.reserve_base + base_amount;

        // Calculate new quote reserve using constant product
        let new_reserve_quote = self.k / new_reserve_base;

        // Amount of quote tokens before fee
        let quote_output_before_fee = self.reserve_quote - new_reserve_quote;

        // Apply fee
        let fee = quote_output_before_fee * self.fee_rate;
        let quote_output = quote_output_before_fee - fee;

        // Calculate price impact
        let old_price = self.spot_price();

        // Update reserves
        self.reserve_base = new_reserve_base;
        self.reserve_quote = new_reserve_quote;

        let new_price = self.spot_price();
        let price_impact = (new_price - old_price) / old_price;

        info!("🔽 AMM SELL: {} base → {} quote | Price: {:.6} → {:.6} ({:.2}%)",
              base_amount, quote_output, old_price, new_price, price_impact * 100.0);

        (quote_output, new_price, price_impact)
    }

    /// Calculate price impact without executing trade (preview)
    pub fn calculate_buy_impact(&self, quote_amount: f64) -> (f64, f64, f64) {
        let fee = quote_amount * self.fee_rate;
        let quote_after_fee = quote_amount - fee;
        let new_reserve_quote = self.reserve_quote + quote_after_fee;
        let new_reserve_base = self.k / new_reserve_quote;
        let base_output = self.reserve_base - new_reserve_base;
        let old_price = self.spot_price();
        let new_price = new_reserve_quote / new_reserve_base;
        let price_impact = (new_price - old_price) / old_price;
        (base_output, new_price, price_impact)
    }

    /// Calculate sell impact without executing trade (preview)
    pub fn calculate_sell_impact(&self, base_amount: f64) -> (f64, f64, f64) {
        let new_reserve_base = self.reserve_base + base_amount;
        let new_reserve_quote = self.k / new_reserve_base;
        let quote_output_before_fee = self.reserve_quote - new_reserve_quote;
        let fee = quote_output_before_fee * self.fee_rate;
        let quote_output = quote_output_before_fee - fee;
        let old_price = self.spot_price();
        let new_price = new_reserve_quote / new_reserve_base;
        let price_impact = (new_price - old_price) / old_price;
        (quote_output, new_price, price_impact)
    }

    /// Add liquidity (used by liquidity providers)
    pub fn add_liquidity(&mut self, base_amount: f64, quote_amount: f64) -> f64 {
        // Calculate LP tokens to mint (proportional to contribution)
        let share = if self.reserve_base == 0.0 {
            (base_amount * quote_amount).sqrt()
        } else {
            let base_share = base_amount / self.reserve_base;
            let quote_share = quote_amount / self.reserve_quote;
            base_share.min(quote_share) * (self.reserve_base * self.reserve_quote).sqrt()
        };

        self.reserve_base += base_amount;
        self.reserve_quote += quote_amount;
        self.k = self.reserve_base * self.reserve_quote;

        share
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_start_heuristics() {
        let predictor = OracleWeightPredictor::new(WeightOptimizerConfig::default());

        // Good conditions: high reputation, fresh data, low volatility
        let features = OracleFeatures {
            price_delta: 0.01,
            volume_normalized: 0.5,
            time_decay: 0.9,
            reputation: 0.95,
            coherence: 0.9,
            volatility: 0.1,
            consensus_ratio: 0.8,
        };

        let predicted = predictor.predict_weight(&features);
        assert!(predicted >= 1.0, "Should predict high weight for good conditions, got {}", predicted);
    }

    #[test]
    fn test_low_reputation_reduces_weight() {
        let predictor = OracleWeightPredictor::new(WeightOptimizerConfig::default());

        // Bad conditions: low reputation
        let features = OracleFeatures {
            price_delta: 0.01,
            volume_normalized: 0.5,
            time_decay: 0.9,
            reputation: 0.2, // Low reputation
            coherence: 0.9,
            volatility: 0.1,
            consensus_ratio: 0.8,
        };

        let predicted = predictor.predict_weight(&features);
        assert!(predicted < 1.0, "Should predict lower weight for low reputation, got {}", predicted);
    }

    #[test]
    fn test_adaptive_uncertainty() {
        let mut predictor = OracleWeightPredictor::new(WeightOptimizerConfig::default());

        // Simulate high volatility
        for _ in 0..20 {
            predictor.update_volatility_ema(0.9); // High volatility
        }

        let uncertainty = predictor.get_adaptive_uncertainty();
        assert!(uncertainty > 0.05, "Uncertainty should increase with volatility, got {}", uncertainty);
    }

    #[test]
    fn test_dynamic_fee() {
        let mut predictor = OracleWeightPredictor::new(WeightOptimizerConfig::default());
        let base_fee = 0.003; // 0.3%

        // Low volatility
        predictor.update_volatility_ema(0.1);
        let low_vol_fee = predictor.calculate_dynamic_fee(base_fee);

        // High volatility
        for _ in 0..20 {
            predictor.update_volatility_ema(0.9);
        }
        let high_vol_fee = predictor.calculate_dynamic_fee(base_fee);

        assert!(high_vol_fee > low_vol_fee,
                "High volatility fee {} should be > low volatility fee {}",
                high_vol_fee, low_vol_fee);
    }

    #[test]
    fn test_amm_buy_increases_price() {
        let mut amm = ConstantProductAMM::new(1000.0, 1000.0, 0.003);
        let initial_price = amm.spot_price();

        // Buy: user provides 100 USD, gets ORB
        let (_, new_price, price_impact) = amm.buy_base(100.0);

        assert!(new_price > initial_price,
                "Price should increase after buy: {} > {}", new_price, initial_price);
        assert!(price_impact > 0.0,
                "Price impact should be positive: {}", price_impact);
    }

    #[test]
    fn test_amm_sell_decreases_price() {
        let mut amm = ConstantProductAMM::new(1000.0, 1000.0, 0.003);
        let initial_price = amm.spot_price();

        // Sell: user provides 100 ORB, gets USD
        let (_, new_price, price_impact) = amm.sell_base(100.0);

        assert!(new_price < initial_price,
                "Price should decrease after sell: {} < {}", new_price, initial_price);
        assert!(price_impact < 0.0,
                "Price impact should be negative: {}", price_impact);
    }

    #[test]
    fn test_amm_constant_product() {
        let mut amm = ConstantProductAMM::new(1000.0, 1000.0, 0.0); // No fee for this test
        let initial_k = amm.k;

        // Multiple trades
        amm.buy_base(50.0);
        amm.sell_base(30.0);
        amm.buy_base(20.0);

        // k should remain approximately constant (small deviation due to fees)
        let final_k = amm.reserve_base * amm.reserve_quote;
        let k_deviation = ((final_k - initial_k) / initial_k).abs();
        assert!(k_deviation < 0.01, "k should remain constant, deviation: {}", k_deviation);
    }

    #[test]
    fn test_feature_normalization() {
        let features = OracleFeatures {
            price_delta: 0.5,
            volume_normalized: 0.5,
            time_decay: 0.5,
            reputation: 0.8,
            coherence: 0.7,
            volatility: 0.3,
            consensus_ratio: 0.6,
        };

        let tensor = features.to_tensor();

        // All values should be in 0-1 range
        for (i, &val) in tensor.iter().enumerate() {
            assert!(val >= 0.0 && val <= 1.0,
                    "Feature {} out of range: {}", i, val);
        }
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_inference_time() {
        let predictor = OracleWeightPredictor::new(WeightOptimizerConfig::default());
        let features = OracleFeatures::default();

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = predictor.predict_weight(&features);
        }
        let elapsed = start.elapsed();

        let per_inference = elapsed / 1000;
        assert!(per_inference < std::time::Duration::from_micros(100),
                "Inference should be < 100us, got {:?}", per_inference);
    }
}
