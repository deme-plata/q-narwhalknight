//! Adaptive ML-Based Confirmation Requirements
//!
//! v1.4.4-beta: Retail-optimized confirmation system with INSTANT FINALITY
//! for small transactions and ML-adaptive security for larger ones.
//!
//! ## Design Philosophy: RETAIL-FIRST
//!
//! For a cryptocurrency to be used at retail (coffee shops, restaurants),
//! customers cannot wait minutes at the counter. We solve this with:
//!
//! 1. **Economic Security Model**: Attack cost > potential theft = safe
//!    - If it costs $54K to attack and you're buying $5 coffee, 0-conf is SAFE
//!    - Attacker profit = $5 - $54,000 = NEGATIVE (irrational attack)
//!
//! 2. **Staking Insurance Pool**: Merchants protected against rare 0-conf attacks
//!    - If double-spend occurs on instant payment, merchant refunded from pool
//!    - Pool funded by small % of staking rewards
//!
//! ## Confirmation Tiers (DAG-Knight Optimistic Finality)
//!
//! DAG-Knight provides FASTER finality than traditional chains because:
//! 1. Multiple blocks produced in parallel (not sequential)
//! 2. Transaction included in DAG immediately upon receipt
//! 3. Probabilistic finality from multiple vertex references
//!
//! | Tier | Confirmations | Time | Value Range | Security Model |
//! |------|---------------|------|-------------|----------------|
//! | ⚡ INSTANT | 0 | <50ms | <$100 | Economic guarantee + mempool inclusion |
//! | 🚀 OPTIMISTIC | 1 | <200ms | <$1K | DAG vertex inclusion (1 producer) |
//! | ✓ FAST | 1 | 2 sec | <$10K | Full block finality |
//! | 🔒 STANDARD | 3 | 6 sec | <$100K | 3-deep DAG confirmation |
//! | 🏦 SETTLEMENT | 30 | 1 min | $100K+ | 30-deep cryptographic proof |
//!
//! For $1M+: Use 300 confirmations (10 minutes) - still 6x faster than Bitcoin!

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Confirmation requirement prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmationRequirement {
    /// Required number of confirmations
    pub confirmations: u64,
    /// Estimated time in seconds
    pub estimated_seconds: u64,
    /// Human-readable time
    pub estimated_time_formatted: String,
    /// Risk level (0.0 - 1.0)
    pub risk_score: f64,
    /// Factors that influenced the decision
    pub factors: Vec<RiskFactor>,
    /// Confidence in the prediction (0.0 - 1.0)
    pub confidence: f64,
}

/// Risk factors that influence confirmation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub weight: f64,
    pub description: String,
}

/// Features used for ML prediction
#[derive(Debug, Clone, Default)]
pub struct ConfirmationFeatures {
    /// Transaction value in USD
    pub tx_value_usd: f64,
    /// Network hashrate in H/s
    pub network_hashrate: u64,
    /// Security bits from cumulative work
    pub security_bits: f64,
    /// Number of connected peers
    pub connected_peers: u64,
    /// Is sender address new (< 10 transactions)?
    pub is_new_sender: bool,
    /// Recent attack attempts detected
    pub recent_attacks: u64,
    /// Current block height
    pub block_height: u64,
    /// Market cap in USD (for security ratio)
    pub market_cap_usd: f64,
}

/// Adaptive confirmation predictor using online linear regression
pub struct AdaptiveConfirmationPredictor {
    /// Model weights (9 features + bias)
    weights: Vec<f64>,
    /// Learning rate for online updates
    learning_rate: f64,
    /// Historical outcomes for learning
    outcomes: VecDeque<ConfirmationOutcome>,
    /// Maximum outcomes to retain
    max_outcomes: usize,
    /// Minimum confirmations (safety floor)
    min_confirmations: u64,
    /// Maximum confirmations (usability ceiling)
    max_confirmations: u64,
    /// Block time in seconds
    block_time_seconds: f64,
    /// Running statistics for feature normalization
    feature_stats: FeatureStats,
}

/// Outcome of a confirmation decision (for learning)
#[derive(Debug, Clone)]
pub struct ConfirmationOutcome {
    pub features: ConfirmationFeatures,
    pub confirmations_used: u64,
    pub was_successful: bool, // No double-spend occurred
    pub timestamp: u64,
}

/// Running statistics for feature normalization
#[derive(Debug, Clone, Default)]
struct FeatureStats {
    count: u64,
    tx_value_mean: f64,
    tx_value_var: f64,
    hashrate_mean: f64,
    hashrate_var: f64,
}

impl Default for AdaptiveConfirmationPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveConfirmationPredictor {
    /// Create a new adaptive confirmation predictor
    pub fn new() -> Self {
        // Initialize weights with domain knowledge priors
        // [tx_value, hashrate, security_bits, peers, new_sender, attacks, height, market_cap, security_ratio, bias]
        let weights = vec![
            0.5,   // tx_value: higher value = more confirmations
            -0.3,  // hashrate: higher hashrate = fewer confirmations (more secure)
            -0.2,  // security_bits: higher security = fewer confirmations
            -0.1,  // peers: more peers = slightly fewer confirmations
            0.2,   // new_sender: new addresses = more confirmations
            0.4,   // recent_attacks: attacks detected = more confirmations
            -0.05, // height: mature chain = slightly fewer confirmations
            0.1,   // market_cap: higher market cap = more at stake
            0.6,   // security_ratio: low security/value ratio = more confirmations
            0.0,   // bias
        ];

        Self {
            weights,
            learning_rate: 0.01,
            outcomes: VecDeque::with_capacity(1000),
            max_outcomes: 1000,
            min_confirmations: 1, // v8.6.0: Reduced from 3 — DAG-Knight parallel finality is sufficient at 1
            max_confirmations: 1000,
            block_time_seconds: 2.0,
            feature_stats: FeatureStats::default(),
        }
    }

    /// Predict required confirmations for a transaction
    pub fn predict(&self, features: &ConfirmationFeatures) -> ConfirmationRequirement {
        // Extract and normalize features
        let feature_vec = self.extract_features(features);

        // Calculate risk score using linear model
        let raw_score: f64 = feature_vec
            .iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        // Apply sigmoid to get risk probability (0-1)
        let risk_score = 1.0 / (1.0 + (-raw_score).exp());

        // Map risk to confirmations using heuristic + ML blend
        let ml_confirmations = self.risk_to_confirmations(risk_score);
        let heuristic_confirmations = self.heuristic_confirmations(features);

        // Blend: 70% heuristic (reliable), 30% ML (adaptive)
        // As we collect more data, we can increase ML weight
        let blend_factor = (self.outcomes.len() as f64 / 500.0).min(0.5);
        let confirmations = (
            heuristic_confirmations as f64 * (1.0 - blend_factor) +
            ml_confirmations as f64 * blend_factor
        ) as u64;

        // Clamp to safety bounds
        let confirmations = confirmations.clamp(self.min_confirmations, self.max_confirmations);

        // Calculate timing
        let estimated_seconds = (confirmations as f64 * self.block_time_seconds) as u64;
        let estimated_time_formatted = Self::format_duration(estimated_seconds);

        // Collect risk factors for transparency
        let factors = self.collect_risk_factors(features);

        // Confidence based on data quantity and feature coverage
        let confidence = self.calculate_confidence(features);

        ConfirmationRequirement {
            confirmations,
            estimated_seconds,
            estimated_time_formatted,
            risk_score,
            factors,
            confidence,
        }
    }

    /// Heuristic-based confirmation calculation (RETAIL-OPTIMIZED)
    fn heuristic_confirmations(&self, features: &ConfirmationFeatures) -> u64 {
        // Calculate economic safety threshold
        // If attack_cost > tx_value * 100, instant payment is economically safe
        let attack_cost = self.estimate_attack_cost(features);
        let instant_threshold = attack_cost / 100.0; // 1% of attack cost
        let quick_threshold = attack_cost / 20.0;    // 5% of attack cost

        // RETAIL-FIRST: Ultra-fast tiered confirmation for DAG-Knight
        // DAG provides faster finality than linear chains!
        let value_confirmations = if features.tx_value_usd < instant_threshold.min(100.0) {
            // ⚡ INSTANT TIER: <50ms (0 confirmations)
            // Coffee, snacks, small retail purchases up to $100
            // Attack cost >> $100, so economically irrational
            // Merchant protected by staking insurance pool
            0
        } else if features.tx_value_usd < quick_threshold.min(1_000.0) {
            // 🚀 OPTIMISTIC TIER: <200ms (1 confirmation)
            // Lunch, retail purchases up to $1K
            // DAG vertex inclusion = probabilistic finality
            1
        } else if features.tx_value_usd < 10_000.0 {
            // ✓ FAST TIER: 2 seconds (1 full block)
            // Electronics, clothing up to $10K
            // Full block with VDF provides strong finality
            1
        } else if features.tx_value_usd < 100_000.0 {
            // 🔒 STANDARD TIER: 6 seconds (3 confirmations)
            // Vehicles, jewelry, high-value items up to $100K
            // 3-deep DAG confirmation
            3
        } else if features.tx_value_usd < 1_000_000.0 {
            // 🏦 SETTLEMENT TIER: 1 minute (30 confirmations)
            // Real estate down payments, large transfers
            30
        } else {
            // 🏛️ INSTITUTIONAL TIER: 10 minutes (300 confirmations)
            // Major real estate, institutional transfers
            // Still 6x FASTER than Bitcoin's 60 minutes!
            300
        };

        // Security adjustment based on network state
        let security_multiplier = if features.security_bits < 40.0 {
            2.0 // Weak security, double confirmations
        } else if features.security_bits < 50.0 {
            1.5 // Moderate security
        } else if features.security_bits < 60.0 {
            1.2 // Good security
        } else {
            1.0 // Strong security
        };

        // Peer adjustment
        let peer_multiplier = if features.connected_peers < 5 {
            1.5 // Few peers, increase confirmations
        } else if features.connected_peers < 20 {
            1.2
        } else {
            1.0
        };

        // New sender adjustment
        let sender_multiplier = if features.is_new_sender { 1.3 } else { 1.0 };

        // Attack detection adjustment
        let attack_multiplier = if features.recent_attacks > 0 {
            1.0 + (features.recent_attacks as f64 * 0.2).min(1.0)
        } else {
            1.0
        };

        // Security ratio check (attack cost vs market cap)
        // If market cap >> attack cost, need more confirmations
        let security_ratio_multiplier = if features.market_cap_usd > 0.0 {
            let safe_market_cap = self.estimate_safe_market_cap(features);
            if features.market_cap_usd > safe_market_cap * 10.0 {
                2.0 // Market cap way exceeds safe threshold
            } else if features.market_cap_usd > safe_market_cap * 2.0 {
                1.5
            } else {
                1.0
            }
        } else {
            1.0
        };

        let total_multiplier = security_multiplier
            * peer_multiplier
            * sender_multiplier
            * attack_multiplier
            * security_ratio_multiplier;

        ((value_confirmations as f64) * total_multiplier) as u64
    }

    /// Estimate attack cost based on network security
    fn estimate_attack_cost(&self, features: &ConfirmationFeatures) -> f64 {
        // Attack cost = 51% hashrate hardware + operating costs
        // RTX 4090: $1,600, 1.5 GH/s SHA3-256
        let hashrate_ghs = features.network_hashrate as f64 / 1e9;
        let attack_hashrate_ghs = hashrate_ghs * 0.51;
        let hardware_cost = attack_hashrate_ghs * 1200.0; // $1,200 per GH/s

        // Include VDF time-lock penalty (2x attack duration)
        let vdf_multiplier = 2.0;

        // Include detection/legal risk premium
        let risk_premium = 1.5;

        hardware_cost * vdf_multiplier * risk_premium
    }

    /// Estimate safe market cap based on network security
    fn estimate_safe_market_cap(&self, features: &ConfirmationFeatures) -> f64 {
        let attack_cost = self.estimate_attack_cost(features);

        // Safe market cap = attack cost * 10 (10% security ratio)
        attack_cost * 10.0
    }

    /// Convert risk score to confirmation count
    fn risk_to_confirmations(&self, risk_score: f64) -> u64 {
        // Exponential mapping: low risk = few confirmations, high risk = many
        let base = 6.0;
        let max_multiplier = 100.0; // Up to 600 confirmations

        let multiplier = 1.0 + (max_multiplier - 1.0) * risk_score.powf(2.0);
        (base * multiplier) as u64
    }

    /// Extract normalized feature vector for ML model
    fn extract_features(&self, features: &ConfirmationFeatures) -> Vec<f64> {
        // Normalize features to 0-1 range
        let tx_value_norm = (features.tx_value_usd.ln().max(0.0) / 20.0).min(1.0); // log scale, cap at ~$500M
        let hashrate_norm = ((features.network_hashrate as f64).ln().max(0.0) / 40.0).min(1.0);
        let security_norm = (features.security_bits / 100.0).min(1.0);
        let peers_norm = (features.connected_peers as f64 / 100.0).min(1.0);
        let new_sender = if features.is_new_sender { 1.0 } else { 0.0 };
        let attacks_norm = (features.recent_attacks as f64 / 10.0).min(1.0);
        let height_norm = ((features.block_height as f64).ln() / 20.0).min(1.0);
        let market_cap_norm = (features.market_cap_usd.ln().max(0.0) / 25.0).min(1.0);

        // Security ratio: attack cost / market cap
        let attack_cost = (features.network_hashrate as f64 / 1e9) * 1200.0 * 0.51;
        let security_ratio = if features.market_cap_usd > 0.0 {
            (attack_cost / features.market_cap_usd).min(1.0)
        } else {
            0.5 // Unknown market cap
        };

        vec![
            tx_value_norm,
            hashrate_norm,
            security_norm,
            peers_norm,
            new_sender,
            attacks_norm,
            height_norm,
            market_cap_norm,
            1.0 - security_ratio, // Invert: low ratio = high risk
            1.0, // Bias term
        ]
    }

    /// Record outcome for online learning
    pub fn record_outcome(&mut self, outcome: ConfirmationOutcome) {
        // Update running statistics
        self.update_feature_stats(&outcome.features);

        // Add to history
        if self.outcomes.len() >= self.max_outcomes {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(outcome.clone());

        // Online gradient descent update
        if outcome.was_successful {
            // If successful with fewer confirmations, we can be more aggressive
            // (reduce weights slightly)
            self.update_weights(&outcome, -0.1);
        } else {
            // If double-spend occurred, we need more confirmations
            // (increase weights significantly)
            self.update_weights(&outcome, 0.5);
            warn!(
                "⚠️ Double-spend detected! Increasing confirmation requirements. \
                TX value: ${:.2}, confirmations used: {}",
                outcome.features.tx_value_usd, outcome.confirmations_used
            );
        }
    }

    /// Update weights using gradient descent
    fn update_weights(&mut self, outcome: &ConfirmationOutcome, direction: f64) {
        let features = self.extract_features(&outcome.features);

        for (i, (w, f)) in self.weights.iter_mut().zip(features.iter()).enumerate() {
            let gradient = direction * f * self.learning_rate;
            *w += gradient;

            // Regularization: keep weights bounded
            *w = w.clamp(-2.0, 2.0);
        }

        debug!(
            "📊 Updated confirmation model weights after {} outcomes",
            self.outcomes.len()
        );
    }

    /// Update feature statistics for normalization
    fn update_feature_stats(&mut self, features: &ConfirmationFeatures) {
        self.feature_stats.count += 1;
        let n = self.feature_stats.count as f64;

        // Welford's online algorithm for mean and variance
        let delta_value = features.tx_value_usd - self.feature_stats.tx_value_mean;
        self.feature_stats.tx_value_mean += delta_value / n;
        let delta2_value = features.tx_value_usd - self.feature_stats.tx_value_mean;
        self.feature_stats.tx_value_var += delta_value * delta2_value;

        let delta_hash = features.network_hashrate as f64 - self.feature_stats.hashrate_mean;
        self.feature_stats.hashrate_mean += delta_hash / n;
        let delta2_hash = features.network_hashrate as f64 - self.feature_stats.hashrate_mean;
        self.feature_stats.hashrate_var += delta_hash * delta2_hash;
    }

    /// Collect risk factors for transparency
    fn collect_risk_factors(&self, features: &ConfirmationFeatures) -> Vec<RiskFactor> {
        let mut factors = Vec::new();

        // Transaction value factor
        let value_weight = match features.tx_value_usd {
            v if v < 100.0 => 0.1,
            v if v < 10_000.0 => 0.3,
            v if v < 100_000.0 => 0.5,
            v if v < 1_000_000.0 => 0.7,
            _ => 1.0,
        };
        factors.push(RiskFactor {
            name: "Transaction Value".to_string(),
            weight: value_weight,
            description: format!("${:.2} transaction", features.tx_value_usd),
        });

        // Network security factor
        let security_weight = if features.security_bits < 45.0 {
            0.8
        } else if features.security_bits < 55.0 {
            0.5
        } else {
            0.2
        };
        factors.push(RiskFactor {
            name: "Network Security".to_string(),
            weight: security_weight,
            description: format!("{:.1} bits cumulative work", features.security_bits),
        });

        // Peer connectivity factor
        if features.connected_peers < 10 {
            factors.push(RiskFactor {
                name: "Low Peer Count".to_string(),
                weight: 0.4,
                description: format!("Only {} peers connected", features.connected_peers),
            });
        }

        // New sender factor
        if features.is_new_sender {
            factors.push(RiskFactor {
                name: "New Sender".to_string(),
                weight: 0.3,
                description: "Address has < 10 previous transactions".to_string(),
            });
        }

        // Recent attacks factor
        if features.recent_attacks > 0 {
            factors.push(RiskFactor {
                name: "Recent Attack Attempts".to_string(),
                weight: 0.6,
                description: format!("{} attacks detected in last 24h", features.recent_attacks),
            });
        }

        // Security ratio factor
        if features.market_cap_usd > 0.0 {
            let safe_cap = self.estimate_safe_market_cap(features);
            if features.market_cap_usd > safe_cap * 2.0 {
                factors.push(RiskFactor {
                    name: "Security Gap".to_string(),
                    weight: 0.7,
                    description: format!(
                        "Market cap ${:.0}M exceeds safe threshold ${:.0}M",
                        features.market_cap_usd / 1e6,
                        safe_cap / 1e6
                    ),
                });
            }
        }

        factors
    }

    /// Calculate confidence in prediction
    fn calculate_confidence(&self, features: &ConfirmationFeatures) -> f64 {
        // Base confidence from data quantity
        let data_confidence = (self.outcomes.len() as f64 / 100.0).min(0.5);

        // Feature coverage confidence
        let mut coverage = 0.0;
        if features.tx_value_usd > 0.0 { coverage += 0.1; }
        if features.network_hashrate > 0 { coverage += 0.1; }
        if features.security_bits > 0.0 { coverage += 0.1; }
        if features.connected_peers > 0 { coverage += 0.1; }
        if features.block_height > 0 { coverage += 0.1; }

        // Heuristic fallback gives baseline confidence
        let baseline_confidence = 0.4;

        (baseline_confidence + data_confidence + coverage).min(0.95)
    }

    /// Format duration in human-readable form
    fn format_duration(seconds: u64) -> String {
        if seconds < 60 {
            format!("{} seconds", seconds)
        } else if seconds < 3600 {
            let mins = seconds / 60;
            let secs = seconds % 60;
            if secs > 0 {
                format!("{} min {} sec", mins, secs)
            } else {
                format!("{} minute{}", mins, if mins > 1 { "s" } else { "" })
            }
        } else {
            let hours = seconds / 3600;
            let mins = (seconds % 3600) / 60;
            format!("{} hour{} {} min", hours, if hours > 1 { "s" } else { "" }, mins)
        }
    }

    /// Get model statistics for monitoring
    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_outcomes": self.outcomes.len(),
            "successful_outcomes": self.outcomes.iter().filter(|o| o.was_successful).count(),
            "failed_outcomes": self.outcomes.iter().filter(|o| !o.was_successful).count(),
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "feature_stats": {
                "sample_count": self.feature_stats.count,
                "tx_value_mean": self.feature_stats.tx_value_mean,
                "hashrate_mean": self.feature_stats.hashrate_mean,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_transaction() {
        let predictor = AdaptiveConfirmationPredictor::new();
        let features = ConfirmationFeatures {
            tx_value_usd: 50.0,
            network_hashrate: 100_000_000, // 100 MH/s
            security_bits: 48.0,
            connected_peers: 10,
            is_new_sender: false,
            recent_attacks: 0,
            block_height: 400_000,
            market_cap_usd: 1_000_000.0,
        };

        let result = predictor.predict(&features);
        assert!(result.confirmations >= 6);
        assert!(result.confirmations <= 30);
        assert!(result.estimated_seconds <= 60);
    }

    #[test]
    fn test_large_transaction() {
        let predictor = AdaptiveConfirmationPredictor::new();
        let features = ConfirmationFeatures {
            tx_value_usd: 1_000_000.0,
            network_hashrate: 100_000_000,
            security_bits: 48.0,
            connected_peers: 10,
            is_new_sender: false,
            recent_attacks: 0,
            block_height: 400_000,
            market_cap_usd: 100_000_000.0,
        };

        let result = predictor.predict(&features);
        assert!(result.confirmations >= 300);
        assert!(result.estimated_seconds >= 600); // At least 10 minutes
    }

    #[test]
    fn test_weak_security_increases_confirmations() {
        let predictor = AdaptiveConfirmationPredictor::new();

        let strong_features = ConfirmationFeatures {
            tx_value_usd: 10_000.0,
            network_hashrate: 1_000_000_000_000, // 1 TH/s
            security_bits: 65.0,
            connected_peers: 50,
            ..Default::default()
        };

        let weak_features = ConfirmationFeatures {
            tx_value_usd: 10_000.0,
            network_hashrate: 100_000_000, // 100 MH/s
            security_bits: 40.0,
            connected_peers: 3,
            ..Default::default()
        };

        let strong_result = predictor.predict(&strong_features);
        let weak_result = predictor.predict(&weak_features);

        assert!(weak_result.confirmations > strong_result.confirmations);
    }
}
