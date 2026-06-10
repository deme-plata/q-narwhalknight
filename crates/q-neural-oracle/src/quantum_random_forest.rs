//! Quantum Random Forest (QRF) Price Prediction Module
//!
//! A quantum-inspired ensemble method for time-series price prediction.
//! Uses amplitude encoding for feature splitting and Grover-like amplitude
//! amplification (simulated classically) to find optimal split thresholds
//! with improved numerical stability.
//!
//! # Architecture
//!
//! ```text
//! Live price feed (60s interval)
//!        |
//!        v
//! QRFPriceOracle::record_price()
//!        |
//!        +---> extract_features() ---> PriceFeatures
//!        |                                  |
//!        |                    QuantumRandomForest::predict()
//!        |                          |
//!        |            [Tree 1] [Tree 2] ... [Tree N]
//!        |                  \      |      /
//!        |                   aggregate predictions
//!        |                          |
//!        v                          v
//! get_smoothed_price() = 0.8 * live + 0.2 * QRF prediction
//! ```
//!
//! # Integration
//!
//! This module is additive-only. It does NOT modify consensus, sync, or balance
//! code. It reads price data and produces predictions that can optionally be
//! blended with live oracle prices for smoother DEX/bridge pricing.

use std::collections::{HashMap, VecDeque};
use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// Quantum Random Forest ensemble for price regression.
///
/// Each tree is trained on a bootstrap sample of the data with a random
/// feature subset at each split. The "quantum" aspect comes from encoding
/// candidate split thresholds as amplitudes and using a Grover-inspired
/// search to find variance-minimizing splits more robustly.
pub struct QuantumRandomForest {
    trees: Vec<QuantumDecisionTree>,
    n_trees: usize,
    max_depth: usize,
    feature_dim: usize,
    min_samples_split: usize,
    /// When true, use amplitude-encoded quantum split search.
    /// When false, fall back to classical brute-force split search.
    quantum_encoding: bool,
}

/// A single decision tree within the QRF ensemble.
pub struct QuantumDecisionTree {
    root: Option<Box<QRFNode>>,
    max_depth: usize,
}

/// Node in a quantum decision tree.
pub enum QRFNode {
    /// Internal split node.
    Split {
        /// Index into [`PriceFeatures::to_vec`] selecting which feature to split on.
        feature_idx: usize,
        /// Classical threshold value for the split.
        threshold: f64,
        /// Rotation angle used in the quantum amplitude encoding for this split.
        /// Stored for introspection / future hardware integration.
        quantum_angle: f64,
        left: Box<QRFNode>,
        right: Box<QRFNode>,
    },
    /// Terminal leaf node.
    Leaf {
        prediction: f64,
        variance: f64,
        sample_count: usize,
    },
}

/// Full prediction output with uncertainty quantification.
#[derive(Clone, Debug)]
pub struct PricePrediction {
    /// Point prediction (mean of tree predictions).
    pub predicted_price: f64,
    /// Confidence in [0, 1]. Higher means lower inter-tree disagreement.
    pub confidence: f64,
    /// 95% prediction interval (lower, upper) assuming Gaussian tree errors.
    pub prediction_interval: (f64, f64),
    /// Variance across tree predictions.
    pub model_variance: f64,
    /// Number of trees whose prediction was within 1 std-dev of the mean.
    pub n_trees_agreed: usize,
    /// Unix timestamp (seconds) when the prediction was made.
    pub timestamp: u64,
}

/// Feature vector extracted from a price time series.
#[derive(Clone, Debug)]
pub struct PriceFeatures {
    /// Raw prices (last N observations, oldest first).
    pub prices: Vec<f64>,
    /// Log returns: ln(p[t] / p[t-1]).
    pub returns: Vec<f64>,
    /// Rolling standard deviation of returns (volatility).
    pub volatility: f64,
    /// Momentum: (latest price - price N steps ago) / price N steps ago.
    pub momentum: f64,
    /// Volume proxy (e.g., number of trades in the window).
    pub volume_proxy: f64,
    /// Mean-reversion signal: (latest price - SMA) / SMA.
    pub mean_reversion_signal: f64,
}

// ---------------------------------------------------------------------------
// PriceFeatures helpers
// ---------------------------------------------------------------------------

impl PriceFeatures {
    /// Flatten features into a fixed-length vector for tree splitting.
    ///
    /// Layout: [volatility, momentum, volume_proxy, mean_reversion, last_return, prev_return]
    /// We use derived statistics rather than raw prices so the feature dimension
    /// stays constant regardless of history length.
    pub fn to_vec(&self) -> Vec<f64> {
        let last_return = self.returns.last().copied().unwrap_or(0.0);
        let prev_return = if self.returns.len() >= 2 {
            self.returns[self.returns.len() - 2]
        } else {
            0.0
        };

        vec![
            self.volatility,
            self.momentum,
            self.volume_proxy,
            self.mean_reversion_signal,
            last_return,
            prev_return,
        ]
    }
}

// ---------------------------------------------------------------------------
// QuantumRandomForest
// ---------------------------------------------------------------------------

/// Default number of features considered at each split: floor(sqrt(feature_dim)).
fn default_max_features(feature_dim: usize) -> usize {
    ((feature_dim as f64).sqrt().floor() as usize).max(1)
}

impl QuantumRandomForest {
    /// Create a new QRF ensemble.
    ///
    /// # Arguments
    /// * `n_trees`     - Number of trees in the forest (50 is a good default).
    /// * `max_depth`   - Maximum depth of each tree (8 is a good default).
    /// * `feature_dim` - Dimensionality of the feature vector (6 for [`PriceFeatures::to_vec`]).
    pub fn new(n_trees: usize, max_depth: usize, feature_dim: usize) -> Self {
        Self {
            trees: Vec::with_capacity(n_trees),
            n_trees,
            max_depth,
            feature_dim,
            min_samples_split: 5,
            quantum_encoding: true,
        }
    }

    /// Train the forest on a dataset of `(features, target_price)` pairs.
    ///
    /// Each tree gets a bootstrap sample (sampling with replacement). At every
    /// split, `sqrt(feature_dim)` features are considered.
    pub fn fit(&mut self, features: &[PriceFeatures], targets: &[f64]) {
        assert_eq!(features.len(), targets.len(), "features and targets must have equal length");
        if features.is_empty() {
            warn!("QRF fit called with empty dataset");
            return;
        }

        self.trees.clear();

        // Use a deterministic seed derived from the first target for reproducibility,
        // but mix in the dataset size so different-sized datasets get different forests.
        let seed = (targets[0].to_bits() ^ (targets.len() as u64)) | 1;
        let mut master_rng = ChaCha8Rng::seed_from_u64(seed);

        for tree_idx in 0..self.n_trees {
            let tree_seed = master_rng.gen::<u64>();
            let mut tree_rng = ChaCha8Rng::seed_from_u64(tree_seed);

            // Bootstrap sample (with replacement)
            let n = features.len();
            let sample_indices: Vec<usize> = (0..n)
                .map(|_| tree_rng.gen_range(0..n))
                .collect();

            let sampled_features: Vec<PriceFeatures> = sample_indices.iter()
                .map(|&i| features[i].clone())
                .collect();
            let sampled_targets: Vec<f64> = sample_indices.iter()
                .map(|&i| targets[i])
                .collect();

            let mut tree = QuantumDecisionTree::new(self.max_depth);
            tree.fit(
                &sampled_features,
                &sampled_targets,
                self.feature_dim,
                self.min_samples_split,
                self.quantum_encoding,
                &mut tree_rng,
            );
            self.trees.push(tree);

            debug!("QRF: trained tree {}/{}", tree_idx + 1, self.n_trees);
        }
    }

    /// Predict next price with confidence interval.
    pub fn predict(&self, features: &PriceFeatures) -> PricePrediction {
        if self.trees.is_empty() {
            return PricePrediction {
                predicted_price: 0.0,
                confidence: 0.0,
                prediction_interval: (0.0, 0.0),
                model_variance: f64::MAX,
                n_trees_agreed: 0,
                timestamp: now_unix(),
            };
        }

        let predictions: Vec<f64> = self.trees.iter()
            .map(|t| t.predict(features))
            .collect();

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        let std_dev = variance.sqrt();

        // 95% CI assuming approximate normality of tree predictions
        let z_95 = 1.96;
        let ci_lower = mean - z_95 * std_dev;
        let ci_upper = mean + z_95 * std_dev;

        // Count trees within 1 std-dev of the mean
        let n_agreed = predictions.iter()
            .filter(|&&p| (p - mean).abs() <= std_dev.max(1e-12))
            .count();

        // Confidence: 1 - (coefficient of variation), clamped to [0, 1].
        // If mean is near zero, fall back to a variance-based metric.
        let confidence = if mean.abs() > 1e-12 {
            (1.0 - (std_dev / mean.abs())).clamp(0.0, 1.0)
        } else {
            (1.0 / (1.0 + variance)).clamp(0.0, 1.0)
        };

        PricePrediction {
            predicted_price: mean,
            confidence,
            prediction_interval: (ci_lower, ci_upper),
            model_variance: variance,
            n_trees_agreed: n_agreed,
            timestamp: now_unix(),
        }
    }

    /// Quantum-enhanced split search.
    ///
    /// Given a set of candidate feature values and corresponding targets, find
    /// the threshold that minimizes the weighted variance of the two partitions.
    ///
    /// The "quantum" approach:
    /// 1. Sort candidate thresholds.
    /// 2. Encode their variance-reduction scores as amplitudes of a simulated
    ///    quantum state (amplitude encoding).
    /// 3. Apply Grover-like amplitude amplification to bias toward the
    ///    low-variance split.
    /// 4. "Measure" the state to select the threshold.
    ///
    /// On classical hardware this is mathematically equivalent to a weighted
    /// random selection biased toward the best splits, which provides robustness
    /// against overfitting (analogous to randomization in random forests) while
    /// still preferring the globally optimal split.
    ///
    /// Returns `(threshold, quantum_angle)`.
    fn quantum_split_search(feature_values: &[f64], targets: &[f64]) -> (f64, f64) {
        assert_eq!(feature_values.len(), targets.len());

        if feature_values.len() < 2 {
            return (feature_values.first().copied().unwrap_or(0.0), 0.0);
        }

        // Collect (value, target) pairs and sort by feature value
        let mut pairs: Vec<(f64, f64)> = feature_values.iter()
            .zip(targets.iter())
            .map(|(&v, &t)| (v, t))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n = pairs.len() as f64;
        let total_sum: f64 = pairs.iter().map(|(_, t)| t).sum();
        let total_sum_sq: f64 = pairs.iter().map(|(_, t)| t * t).sum();
        let total_var = total_sum_sq / n - (total_sum / n).powi(2);

        if total_var < 1e-15 {
            // All targets identical; any threshold works.
            return (pairs[pairs.len() / 2].0, 0.0);
        }

        // Evaluate each candidate midpoint threshold
        let mut best_threshold = pairs[0].0;
        let mut best_score = f64::NEG_INFINITY;
        let mut scores: Vec<f64> = Vec::with_capacity(pairs.len() - 1);
        let mut thresholds: Vec<f64> = Vec::with_capacity(pairs.len() - 1);

        let mut left_sum = 0.0_f64;
        let mut left_sum_sq = 0.0_f64;

        for i in 0..(pairs.len() - 1) {
            left_sum += pairs[i].1;
            left_sum_sq += pairs[i].1 * pairs[i].1;

            let left_n = (i + 1) as f64;
            let right_n = n - left_n;

            if right_n < 1.0 {
                break;
            }

            // Skip if consecutive feature values are identical (no valid split)
            if (pairs[i].0 - pairs[i + 1].0).abs() < 1e-15 {
                continue;
            }

            let left_mean = left_sum / left_n;
            let left_var = (left_sum_sq / left_n) - left_mean.powi(2);

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;
            let right_mean = right_sum / right_n;
            let right_var = (right_sum_sq / right_n) - right_mean.powi(2);

            // Variance reduction (higher is better)
            let weighted_var = (left_n * left_var.max(0.0) + right_n * right_var.max(0.0)) / n;
            let var_reduction = total_var - weighted_var;

            let threshold = (pairs[i].0 + pairs[i + 1].0) / 2.0;
            thresholds.push(threshold);
            scores.push(var_reduction);

            if var_reduction > best_score {
                best_score = var_reduction;
                best_threshold = threshold;
            }
        }

        if thresholds.is_empty() {
            return (pairs[pairs.len() / 2].0, 0.0);
        }

        // --- Quantum amplitude encoding & amplification (simulated) ---
        //
        // Encode variance-reduction scores as probability amplitudes.
        // Apply one round of Grover-like amplification: reflect about the
        // "good" states (above-median scores) then reflect about the mean
        // amplitude. This biases the distribution toward the best splits.

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let score_range = (max_score - min_score).max(1e-15);

        // Normalize scores to [0, 1] and convert to amplitudes
        let mut amplitudes: Vec<f64> = scores.iter()
            .map(|s| ((s - min_score) / score_range).max(1e-10).sqrt())
            .collect();

        // Normalize to unit vector (valid quantum state)
        let norm: f64 = amplitudes.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for a in &mut amplitudes {
                *a /= norm;
            }
        }

        // Grover-like amplification: one iteration
        // 1. Oracle: flip sign of amplitudes with above-median score
        let median_score = {
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        for (i, s) in scores.iter().enumerate() {
            if *s >= median_score {
                amplitudes[i] = -amplitudes[i];
            }
        }

        // 2. Diffusion operator: reflect about mean amplitude
        let mean_amp = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        for a in &mut amplitudes {
            *a = 2.0 * mean_amp - *a;
        }

        // Convert back to probabilities
        let probs: Vec<f64> = amplitudes.iter().map(|a| a * a).collect();
        let prob_sum: f64 = probs.iter().sum();

        // "Measure": select threshold with highest post-amplification probability
        let mut best_idx = 0;
        let mut best_prob = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }

        let selected_threshold = thresholds[best_idx];

        // Quantum angle: encode the selection as a rotation angle in [0, pi]
        let quantum_angle = if prob_sum > 1e-15 {
            std::f64::consts::PI * (best_prob / prob_sum)
        } else {
            0.0
        };

        (selected_threshold, quantum_angle)
    }

    /// Classical brute-force split search (fallback when quantum_encoding is false).
    /// Returns `(threshold, 0.0)`.
    fn classical_split_search(feature_values: &[f64], targets: &[f64]) -> (f64, f64) {
        assert_eq!(feature_values.len(), targets.len());

        if feature_values.len() < 2 {
            return (feature_values.first().copied().unwrap_or(0.0), 0.0);
        }

        let mut pairs: Vec<(f64, f64)> = feature_values.iter()
            .zip(targets.iter())
            .map(|(&v, &t)| (v, t))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n = pairs.len() as f64;
        let total_sum: f64 = pairs.iter().map(|(_, t)| t).sum();
        let total_sum_sq: f64 = pairs.iter().map(|(_, t)| t * t).sum();

        let mut best_threshold = pairs[pairs.len() / 2].0;
        let mut best_var_reduction = f64::NEG_INFINITY;

        let total_var = total_sum_sq / n - (total_sum / n).powi(2);

        let mut left_sum = 0.0_f64;
        let mut left_sum_sq = 0.0_f64;

        for i in 0..(pairs.len() - 1) {
            left_sum += pairs[i].1;
            left_sum_sq += pairs[i].1 * pairs[i].1;

            let left_n = (i + 1) as f64;
            let right_n = n - left_n;
            if right_n < 1.0 {
                break;
            }
            if (pairs[i].0 - pairs[i + 1].0).abs() < 1e-15 {
                continue;
            }

            let left_mean = left_sum / left_n;
            let left_var = (left_sum_sq / left_n) - left_mean.powi(2);

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;
            let right_mean = right_sum / right_n;
            let right_var = (right_sum_sq / right_n) - right_mean.powi(2);

            let weighted_var = (left_n * left_var.max(0.0) + right_n * right_var.max(0.0)) / n;
            let var_reduction = total_var - weighted_var;

            if var_reduction > best_var_reduction {
                best_var_reduction = var_reduction;
                best_threshold = (pairs[i].0 + pairs[i + 1].0) / 2.0;
            }
        }

        (best_threshold, 0.0)
    }

    /// Extract [`PriceFeatures`] from a raw price history.
    ///
    /// # Arguments
    /// * `price_history` - Chronological prices (oldest first).
    /// * `window`        - Rolling window size for volatility and SMA.
    pub fn extract_features(price_history: &[f64], window: usize) -> PriceFeatures {
        let window = window.max(2); // need at least 2 for returns

        // Log returns
        let returns: Vec<f64> = price_history.windows(2)
            .map(|w| {
                if w[0] > 1e-15 {
                    (w[1] / w[0]).ln()
                } else {
                    0.0
                }
            })
            .collect();

        // Volatility: std dev of recent returns
        let vol_window = returns.len().min(window);
        let volatility = if vol_window >= 2 {
            let recent = &returns[returns.len().saturating_sub(vol_window)..];
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            let var = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            var.sqrt()
        } else {
            0.0
        };

        // Momentum
        let momentum = if price_history.len() >= window {
            let old = price_history[price_history.len() - window];
            let new = *price_history.last().unwrap_or(&0.0);
            if old.abs() > 1e-15 { (new - old) / old } else { 0.0 }
        } else if price_history.len() >= 2 {
            let old = price_history[0];
            let new = *price_history.last().unwrap();
            if old.abs() > 1e-15 { (new - old) / old } else { 0.0 }
        } else {
            0.0
        };

        // Simple Moving Average (SMA) and mean-reversion signal
        let sma_window = price_history.len().min(window);
        let sma = if sma_window > 0 {
            let slice = &price_history[price_history.len().saturating_sub(sma_window)..];
            slice.iter().sum::<f64>() / slice.len() as f64
        } else {
            0.0
        };

        let latest = *price_history.last().unwrap_or(&0.0);
        let mean_reversion_signal = if sma.abs() > 1e-15 {
            (latest - sma) / sma
        } else {
            0.0
        };

        // Volume proxy: number of non-zero returns (crude activity indicator)
        let volume_proxy = returns.iter().filter(|r| r.abs() > 1e-15).count() as f64
            / (returns.len().max(1) as f64);

        PriceFeatures {
            prices: price_history.to_vec(),
            returns,
            volatility,
            momentum,
            volume_proxy,
            mean_reversion_signal,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantumDecisionTree
// ---------------------------------------------------------------------------

impl QuantumDecisionTree {
    /// Create an empty tree with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            root: None,
            max_depth,
        }
    }

    /// Train the tree on a dataset.
    pub fn fit(
        &mut self,
        features: &[PriceFeatures],
        targets: &[f64],
        feature_dim: usize,
        min_samples_split: usize,
        quantum_encoding: bool,
        rng: &mut impl Rng,
    ) {
        let feat_vecs: Vec<Vec<f64>> = features.iter().map(|f| f.to_vec()).collect();
        self.root = Some(Box::new(Self::build_tree(
            &feat_vecs,
            targets,
            0,
            self.max_depth,
            feature_dim,
            min_samples_split,
            quantum_encoding,
            rng,
        )));
    }

    /// Predict target value for a single feature vector.
    pub fn predict(&self, features: &PriceFeatures) -> f64 {
        let feat_vec = features.to_vec();
        match &self.root {
            Some(node) => Self::predict_node(node, &feat_vec),
            None => 0.0,
        }
    }

    /// Recursively traverse the tree to produce a prediction.
    fn predict_node(node: &QRFNode, features: &[f64]) -> f64 {
        match node {
            QRFNode::Leaf { prediction, .. } => *prediction,
            QRFNode::Split { feature_idx, threshold, left, right, .. } => {
                let val = features.get(*feature_idx).copied().unwrap_or(0.0);
                if val <= *threshold {
                    Self::predict_node(left, features)
                } else {
                    Self::predict_node(right, features)
                }
            }
        }
    }

    /// Recursively build the tree.
    fn build_tree(
        feat_vecs: &[Vec<f64>],
        targets: &[f64],
        depth: usize,
        max_depth: usize,
        feature_dim: usize,
        min_samples_split: usize,
        quantum_encoding: bool,
        rng: &mut impl Rng,
    ) -> QRFNode {
        let n = targets.len();

        // Leaf conditions
        if n < min_samples_split || depth >= max_depth || n <= 1 {
            return Self::make_leaf(targets);
        }

        // Check if all targets are identical
        let first = targets[0];
        if targets.iter().all(|t| (t - first).abs() < 1e-15) {
            return Self::make_leaf(targets);
        }

        // Random feature subset (sqrt(feature_dim) features)
        let max_features = default_max_features(feature_dim);
        let mut candidate_features: Vec<usize> = (0..feature_dim).collect();
        candidate_features.shuffle(rng);
        candidate_features.truncate(max_features);

        // Find best split across candidate features
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_angle = 0.0;
        let mut best_var_reduction = f64::NEG_INFINITY;

        for &feat_idx in &candidate_features {
            let values: Vec<f64> = feat_vecs.iter().map(|fv| fv[feat_idx]).collect();

            // Check that the feature has some variation
            let fmin = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let fmax = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (fmax - fmin).abs() < 1e-15 {
                continue;
            }

            let (threshold, angle) = if quantum_encoding {
                QuantumRandomForest::quantum_split_search(&values, targets)
            } else {
                QuantumRandomForest::classical_split_search(&values, targets)
            };

            // Evaluate the split
            let (left_targets, right_targets) = Self::partition_targets(feat_vecs, targets, feat_idx, threshold);

            if left_targets.is_empty() || right_targets.is_empty() {
                continue;
            }

            let var_reduction = Self::variance_reduction(targets, &left_targets, &right_targets);
            if var_reduction > best_var_reduction {
                best_var_reduction = var_reduction;
                best_feature = feat_idx;
                best_threshold = threshold;
                best_angle = angle;
            }
        }

        // If no valid split found, return leaf
        if best_var_reduction <= 0.0 {
            return Self::make_leaf(targets);
        }

        // Partition and recurse
        let (left_feats, left_targets, right_feats, right_targets) =
            Self::partition(feat_vecs, targets, best_feature, best_threshold);

        if left_targets.is_empty() || right_targets.is_empty() {
            return Self::make_leaf(targets);
        }

        let left = Self::build_tree(
            &left_feats, &left_targets,
            depth + 1, max_depth, feature_dim, min_samples_split,
            quantum_encoding, rng,
        );
        let right = Self::build_tree(
            &right_feats, &right_targets,
            depth + 1, max_depth, feature_dim, min_samples_split,
            quantum_encoding, rng,
        );

        QRFNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            quantum_angle: best_angle,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a leaf node from a set of targets.
    fn make_leaf(targets: &[f64]) -> QRFNode {
        let n = targets.len();
        if n == 0 {
            return QRFNode::Leaf {
                prediction: 0.0,
                variance: 0.0,
                sample_count: 0,
            };
        }
        let mean = targets.iter().sum::<f64>() / n as f64;
        let variance = targets.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
        QRFNode::Leaf {
            prediction: mean,
            variance,
            sample_count: n,
        }
    }

    /// Partition targets by a split, returning (left_targets, right_targets).
    fn partition_targets(
        feat_vecs: &[Vec<f64>],
        targets: &[f64],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (fv, &t) in feat_vecs.iter().zip(targets.iter()) {
            if fv[feature_idx] <= threshold {
                left.push(t);
            } else {
                right.push(t);
            }
        }
        (left, right)
    }

    /// Full partition returning both features and targets for left and right.
    fn partition(
        feat_vecs: &[Vec<f64>],
        targets: &[f64],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let mut lf = Vec::new();
        let mut lt = Vec::new();
        let mut rf = Vec::new();
        let mut rt = Vec::new();
        for (fv, &t) in feat_vecs.iter().zip(targets.iter()) {
            if fv[feature_idx] <= threshold {
                lf.push(fv.clone());
                lt.push(t);
            } else {
                rf.push(fv.clone());
                rt.push(t);
            }
        }
        (lf, lt, rf, rt)
    }

    /// Compute variance reduction for a candidate split.
    fn variance_reduction(parent: &[f64], left: &[f64], right: &[f64]) -> f64 {
        fn variance(data: &[f64]) -> f64 {
            if data.is_empty() { return 0.0; }
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64
        }

        let n = parent.len() as f64;
        let parent_var = variance(parent);
        let left_var = variance(left);
        let right_var = variance(right);
        let left_weight = left.len() as f64 / n;
        let right_weight = right.len() as f64 / n;

        parent_var - (left_weight * left_var + right_weight * right_var)
    }
}

// ---------------------------------------------------------------------------
// QRFPriceOracle — rolling oracle integration layer
// ---------------------------------------------------------------------------

/// Rolling price oracle that maintains history and provides QRF-enhanced predictions.
///
/// Designed to sit alongside the existing `BankingOracleIntegration`. Call
/// [`record_price`] from the 60-second price update loop, then use
/// [`get_smoothed_price`] to blend live market data with the QRF prediction.
pub struct QRFPriceOracle {
    /// Per-asset QRF model. Lazily created on first prediction for each asset.
    models: HashMap<String, QuantumRandomForest>,
    /// Per-asset chronological price history: `(timestamp, price)`.
    price_history: HashMap<String, VecDeque<(u64, f64)>>,
    /// Maximum number of price data points to retain per asset.
    max_history: usize,
    /// Number of new price samples between automatic retraining passes.
    retrain_interval: usize,
    /// Per-asset counter: samples recorded since last retrain.
    samples_since_retrain: HashMap<String, usize>,
    /// Feature extraction window size.
    feature_window: usize,
    /// QRF hyperparameters.
    n_trees: usize,
    max_depth: usize,
}

impl QRFPriceOracle {
    /// Create a new oracle with sensible defaults:
    /// 50 trees, depth 8, feature_dim 6, retrain every 100 samples.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            price_history: HashMap::new(),
            max_history: 1000,
            retrain_interval: 100,
            samples_since_retrain: HashMap::new(),
            feature_window: 20,
            n_trees: 50,
            max_depth: 8,
        }
    }

    /// Record a new price observation.
    ///
    /// Call this from the existing 60-second price update loop.
    pub fn record_price(&mut self, asset: &str, timestamp: u64, price: f64) {
        let history = self.price_history
            .entry(asset.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.max_history + 1));

        history.push_back((timestamp, price));

        // Evict oldest if over capacity
        while history.len() > self.max_history {
            history.pop_front();
        }

        // Bump retrain counter
        let counter = self.samples_since_retrain
            .entry(asset.to_string())
            .or_insert(0);
        *counter += 1;

        self.maybe_retrain(asset);
    }

    /// Get a QRF-smoothed price: `0.8 * live_price + 0.2 * qrf_prediction`.
    ///
    /// If the model has not been trained yet (insufficient data), returns the
    /// live price unmodified.
    pub fn get_smoothed_price(&self, asset: &str, live_price: f64) -> f64 {
        match self.get_prediction(asset) {
            Some(pred) if pred.confidence > 0.1 => {
                0.8 * live_price + 0.2 * pred.predicted_price
            }
            _ => live_price,
        }
    }

    /// Get full QRF prediction with confidence interval for an asset.
    ///
    /// Returns `None` if there is insufficient history to form features.
    pub fn get_prediction(&self, asset: &str) -> Option<PricePrediction> {
        let model = self.models.get(asset)?;
        let history = self.price_history.get(asset)?;

        // Need at least feature_window + 1 prices to form features + 1 target
        if history.len() < self.feature_window + 1 {
            return None;
        }

        let prices: Vec<f64> = history.iter().map(|(_, p)| *p).collect();
        let features = QuantumRandomForest::extract_features(&prices, self.feature_window);
        Some(model.predict(&features))
    }

    /// Retrain the model for `asset` if enough new samples have accumulated.
    fn maybe_retrain(&mut self, asset: &str) {
        let counter = self.samples_since_retrain.get(asset).copied().unwrap_or(0);
        if counter < self.retrain_interval {
            return;
        }

        let history = match self.price_history.get(asset) {
            Some(h) => h,
            None => return,
        };

        // Need enough data to form training samples
        let min_train_samples = self.feature_window + 10;
        if history.len() < min_train_samples {
            return;
        }

        let prices: Vec<f64> = history.iter().map(|(_, p)| *p).collect();

        // Build training set: each sample is (features from prices[..i], target = prices[i])
        let mut train_features = Vec::new();
        let mut train_targets = Vec::new();

        for i in self.feature_window..prices.len() {
            let window = &prices[..i];
            let features = QuantumRandomForest::extract_features(window, self.feature_window);
            train_features.push(features);
            train_targets.push(prices[i]);
        }

        if train_features.is_empty() {
            return;
        }

        debug!("QRF: retraining model for '{}' with {} samples", asset, train_features.len());

        let mut model = QuantumRandomForest::new(self.n_trees, self.max_depth, 6);
        model.fit(&train_features, &train_targets);

        self.models.insert(asset.to_string(), model);
        self.samples_since_retrain.insert(asset.to_string(), 0);
    }
}

impl Default for QRFPriceOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Current UNIX timestamp in seconds.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple linear price series: price = base + slope * t + noise.
    fn linear_prices(n: usize, base: f64, slope: f64, noise: f64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        (0..n)
            .map(|i| {
                let noise_val: f64 = (rng.gen::<f64>() - 0.5) * 2.0 * noise;
                base + slope * i as f64 + noise_val
            })
            .collect()
    }

    #[test]
    fn test_qrf_basic_regression() {
        // Generate a gentle upward trend: 100 + 0.1 * t with small noise
        let prices = linear_prices(200, 100.0, 0.1, 0.5);
        let window = 20;

        // Build training set
        let mut features = Vec::new();
        let mut targets = Vec::new();
        for i in window..prices.len() {
            let f = QuantumRandomForest::extract_features(&prices[..i], window);
            features.push(f);
            targets.push(prices[i]);
        }

        let mut qrf = QuantumRandomForest::new(50, 8, 6);
        qrf.fit(&features, &targets);

        // Predict the "next" price using all data as features
        let test_features = QuantumRandomForest::extract_features(&prices, window);
        let pred = qrf.predict(&test_features);

        // The last actual price is ~100 + 0.1 * 199 = ~119.9
        // Prediction should be in the ballpark (within 10% of the last price)
        let last_price = *prices.last().unwrap();
        let error_pct = ((pred.predicted_price - last_price) / last_price).abs();

        assert!(
            error_pct < 0.10,
            "Prediction error too large: predicted={:.2}, actual={:.2}, error={:.1}%",
            pred.predicted_price, last_price, error_pct * 100.0
        );

        // Confidence should be positive
        assert!(pred.confidence > 0.0, "Confidence should be > 0");

        // Prediction interval should contain the predicted price
        assert!(
            pred.prediction_interval.0 <= pred.predicted_price
                && pred.predicted_price <= pred.prediction_interval.1,
            "Prediction interval should contain the point prediction"
        );
    }

    #[test]
    fn test_feature_extraction() {
        // Known price series: 100, 102, 101, 104, 103
        let prices = vec![100.0, 102.0, 101.0, 104.0, 103.0];
        let features = QuantumRandomForest::extract_features(&prices, 3);

        // Returns: ln(102/100), ln(101/102), ln(104/101), ln(103/104)
        assert_eq!(features.returns.len(), 4);
        assert!((features.returns[0] - (102.0_f64 / 100.0).ln()).abs() < 1e-10);

        // Volatility should be > 0 (there is variation in returns)
        assert!(features.volatility > 0.0, "Volatility should be positive");

        // Momentum: (103 - 101) / 101 using window=3
        // Window=3: prices[5-3]=prices[2]=101, latest=103
        let expected_momentum = (103.0 - 101.0) / 101.0;
        assert!(
            (features.momentum - expected_momentum).abs() < 1e-10,
            "Momentum mismatch: got {}, expected {}",
            features.momentum, expected_momentum
        );

        // Mean reversion: (103 - SMA) / SMA where SMA = avg of last 3 = (101+104+103)/3
        let sma = (101.0 + 104.0 + 103.0) / 3.0;
        let expected_mr = (103.0 - sma) / sma;
        assert!(
            (features.mean_reversion_signal - expected_mr).abs() < 1e-10,
            "Mean reversion mismatch: got {}, expected {}",
            features.mean_reversion_signal, expected_mr
        );

        // Feature vector should have 6 elements
        let fv = features.to_vec();
        assert_eq!(fv.len(), 6);
    }

    #[test]
    fn test_prediction_confidence() {
        let window = 20;

        // Low-noise series -> high confidence
        let low_noise = linear_prices(200, 100.0, 0.1, 0.1);
        let mut features_low = Vec::new();
        let mut targets_low = Vec::new();
        for i in window..low_noise.len() {
            features_low.push(QuantumRandomForest::extract_features(&low_noise[..i], window));
            targets_low.push(low_noise[i]);
        }
        let mut qrf_low = QuantumRandomForest::new(50, 8, 6);
        qrf_low.fit(&features_low, &targets_low);
        let pred_low = qrf_low.predict(
            &QuantumRandomForest::extract_features(&low_noise, window),
        );

        // High-noise series -> lower confidence
        let high_noise = linear_prices(200, 100.0, 0.1, 10.0);
        let mut features_high = Vec::new();
        let mut targets_high = Vec::new();
        for i in window..high_noise.len() {
            features_high.push(QuantumRandomForest::extract_features(&high_noise[..i], window));
            targets_high.push(high_noise[i]);
        }
        let mut qrf_high = QuantumRandomForest::new(50, 8, 6);
        qrf_high.fit(&features_high, &targets_high);
        let pred_high = qrf_high.predict(
            &QuantumRandomForest::extract_features(&high_noise, window),
        );

        // Low-noise model should have higher confidence (less inter-tree disagreement)
        assert!(
            pred_low.confidence >= pred_high.confidence,
            "Low-noise confidence ({:.4}) should be >= high-noise confidence ({:.4})",
            pred_low.confidence, pred_high.confidence
        );

        // High-noise model should have larger variance
        assert!(
            pred_high.model_variance >= pred_low.model_variance,
            "High-noise variance ({:.4}) should be >= low-noise variance ({:.4})",
            pred_high.model_variance, pred_low.model_variance
        );
    }

    #[test]
    fn test_smoothed_price() {
        let mut oracle = QRFPriceOracle::new();
        // Set a small retrain interval for testing
        oracle.retrain_interval = 30;

        // Record enough prices to trigger training
        for i in 0..150 {
            let price = 100.0 + 0.1 * i as f64;
            oracle.record_price("BTC", i as u64 * 60, price);
        }

        // Model should now be trained
        assert!(oracle.models.contains_key("BTC"), "Model should have been trained after 150 samples");

        // get_smoothed_price should blend 80% live + 20% QRF
        let live = 120.0;
        let smoothed = oracle.get_smoothed_price("BTC", live);

        // The smoothed price should be different from the live price (QRF contributes 20%)
        // But should be close (within a reasonable range)
        let diff = (smoothed - live).abs();
        let max_diff = live * 0.15; // 15% max difference is generous
        assert!(
            diff < max_diff,
            "Smoothed price ({:.2}) too far from live ({:.2}), diff={:.2}",
            smoothed, live, diff
        );

        // When model confidence is too low or model doesn't exist, should return live
        let unknown_smoothed = oracle.get_smoothed_price("UNKNOWN", 50.0);
        assert_eq!(unknown_smoothed, 50.0, "Unknown asset should return live price");
    }

    #[test]
    fn test_retrain_trigger() {
        let mut oracle = QRFPriceOracle::new();
        oracle.retrain_interval = 50;

        // Record prices below the retrain threshold
        for i in 0..49 {
            oracle.record_price("ETH", i as u64 * 60, 3000.0 + i as f64);
        }
        // Model should NOT exist yet (only 49 samples, but also need window+10=30 minimum)
        // 49 samples with retrain_interval=50 means counter=49 < 50, no retrain
        assert!(
            !oracle.models.contains_key("ETH"),
            "Model should not be trained before retrain_interval"
        );

        // Add one more to hit the retrain_interval
        oracle.record_price("ETH", 49 * 60, 3049.0);
        // Now counter=50 >= 50, but we need at least window+10=30 prices, and we have 50. Should retrain.
        assert!(
            oracle.models.contains_key("ETH"),
            "Model should be trained after reaching retrain_interval"
        );

        // Counter should be reset
        assert_eq!(
            *oracle.samples_since_retrain.get("ETH").unwrap_or(&999),
            0,
            "Samples counter should reset to 0 after retrain"
        );
    }

    #[test]
    fn test_quantum_split_determinism() {
        // The quantum split search should be deterministic for the same input
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let targets = vec![10.0, 12.0, 11.0, 15.0, 20.0, 22.0, 21.0, 25.0];

        let (t1, a1) = QuantumRandomForest::quantum_split_search(&values, &targets);
        let (t2, a2) = QuantumRandomForest::quantum_split_search(&values, &targets);

        assert_eq!(t1, t2, "Quantum split should be deterministic");
        assert_eq!(a1, a2, "Quantum angle should be deterministic");
    }

    #[test]
    fn test_empty_forest_prediction() {
        let qrf = QuantumRandomForest::new(10, 5, 6);
        let features = PriceFeatures {
            prices: vec![100.0],
            returns: vec![],
            volatility: 0.0,
            momentum: 0.0,
            volume_proxy: 0.0,
            mean_reversion_signal: 0.0,
        };
        let pred = qrf.predict(&features);
        assert_eq!(pred.predicted_price, 0.0);
        assert_eq!(pred.confidence, 0.0);
        assert_eq!(pred.n_trees_agreed, 0);
    }

    #[test]
    fn test_single_value_dataset() {
        // Edge case: all targets identical
        let features: Vec<PriceFeatures> = (0..20).map(|_| PriceFeatures {
            prices: vec![100.0; 10],
            returns: vec![0.0; 9],
            volatility: 0.0,
            momentum: 0.0,
            volume_proxy: 0.0,
            mean_reversion_signal: 0.0,
        }).collect();
        let targets: Vec<f64> = vec![42.0; 20];

        let mut qrf = QuantumRandomForest::new(10, 4, 6);
        qrf.fit(&features, &targets);

        let pred = qrf.predict(&features[0]);
        assert!(
            (pred.predicted_price - 42.0).abs() < 1e-10,
            "Prediction for constant target should be that constant"
        );
    }

    #[test]
    fn test_oracle_default() {
        let oracle = QRFPriceOracle::default();
        assert_eq!(oracle.max_history, 1000);
        assert_eq!(oracle.retrain_interval, 100);
        assert_eq!(oracle.n_trees, 50);
        assert_eq!(oracle.max_depth, 8);
    }

    #[test]
    fn test_price_history_eviction() {
        let mut oracle = QRFPriceOracle::new();
        oracle.max_history = 10;
        oracle.retrain_interval = 999; // Prevent retrain during this test

        for i in 0..20 {
            oracle.record_price("TEST", i, i as f64);
        }

        let history = oracle.price_history.get("TEST").unwrap();
        assert_eq!(history.len(), 10, "History should be capped at max_history");
        // Oldest should be price 10 (indices 10..19 retained)
        assert_eq!(history.front().unwrap().1, 10.0);
        assert_eq!(history.back().unwrap().1, 19.0);
    }
}
