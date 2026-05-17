//! Oracle Fallback System
//!
//! Multi-layer fallback for oracle failures.

use crate::{Prediction, PredictionContext, PredictionSource};
use crate::experts::PredictionDomain;
use std::collections::{HashMap, VecDeque};
use tracing::{debug, warn};

/// Multi-layer fallback system for oracle failures
pub struct OracleFallbackSystem {
    /// Time-weighted moving average fallback
    twma: TWMAFallback,

    /// Simple heuristic fallback
    heuristic: HeuristicFallback,

    /// Health monitor
    health: FallbackHealth,
}

impl OracleFallbackSystem {
    /// Create new fallback system
    pub fn new() -> Self {
        Self {
            twma: TWMAFallback::new(100), // 100 sample window
            heuristic: HeuristicFallback::new(),
            health: FallbackHealth::default(),
        }
    }

    /// Get prediction using fallback chain
    pub async fn get_prediction(
        &self,
        domain: PredictionDomain,
        context: &PredictionContext,
    ) -> anyhow::Result<Prediction> {
        // Try TWMA first (if we have history)
        if let Some(twma_pred) = self.twma.calculate(domain) {
            debug!("Using TWMA fallback for {:?}", domain);
            return Ok(Prediction {
                value: twma_pred,
                confidence: 0.4, // Lower confidence for TWMA
                domain,
                source: PredictionSource::TWMA,
                expert_weights: vec![],
                quantum_fidelity: 0.0,
                timestamp: chrono::Utc::now().timestamp() as u64,
                proof: None,
            });
        }

        // Ultimate fallback: heuristics
        debug!("Using heuristic fallback for {:?}", domain);
        let heuristic_pred = self.heuristic.calculate(domain, context);

        Ok(Prediction {
            value: heuristic_pred,
            confidence: 0.3, // Lowest confidence
            domain,
            source: PredictionSource::Fallback,
            expert_weights: vec![],
            quantum_fidelity: 0.0,
            timestamp: chrono::Utc::now().timestamp() as u64,
            proof: None,
        })
    }

    /// Record a successful prediction for TWMA
    pub fn record_prediction(&mut self, domain: PredictionDomain, value: f64) {
        self.twma.record(domain, value);
    }
}

impl Default for OracleFallbackSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Time-Weighted Moving Average fallback
struct TWMAFallback {
    /// Historical values per domain
    history: HashMap<PredictionDomain, VecDeque<HistoricalValue>>,

    /// Window size
    window_size: usize,
}

#[derive(Clone, Debug)]
struct HistoricalValue {
    value: f64,
    timestamp: u64,
}

impl TWMAFallback {
    fn new(window_size: usize) -> Self {
        Self {
            history: HashMap::new(),
            window_size,
        }
    }

    fn record(&mut self, domain: PredictionDomain, value: f64) {
        let history = self.history.entry(domain).or_insert_with(VecDeque::new);

        history.push_back(HistoricalValue {
            value,
            timestamp: chrono::Utc::now().timestamp() as u64,
        });

        // Limit size
        while history.len() > self.window_size {
            history.pop_front();
        }
    }

    fn calculate(&self, domain: PredictionDomain) -> Option<f64> {
        let history = self.history.get(&domain)?;

        if history.is_empty() {
            return None;
        }

        let now = chrono::Utc::now().timestamp() as u64;
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for entry in history {
            let age = now.saturating_sub(entry.timestamp);
            // Exponential decay with 24-hour half-life
            let weight = (-(age as f64) / (24.0 * 3600.0)).exp();

            weighted_sum += entry.value * weight;
            weight_sum += weight;
        }

        if weight_sum > 1e-10 {
            Some(weighted_sum / weight_sum)
        } else {
            None
        }
    }
}

/// Simple heuristic fallback
struct HeuristicFallback;

impl HeuristicFallback {
    fn new() -> Self {
        Self
    }

    fn calculate(&self, domain: PredictionDomain, context: &PredictionContext) -> f64 {
        match domain {
            PredictionDomain::FeeForecasting => {
                // Simple fee heuristic: based on transaction volume
                let base_fee = 1.0;
                let volume_factor = (context.tx_volume / 10_000.0).ln().max(0.0) / 10.0;
                (base_fee + volume_factor).min(10.0)
            }

            PredictionDomain::VDFOptimization => {
                // VDF difficulty based on block height
                let epoch = context.block_height / 10_000;
                0.5 + (epoch as f64 * 0.001).min(0.4)
            }

            PredictionDomain::ReserveManagement => {
                // Reserve target: 10-30% of staking pool
                let staking_ratio = context.staking_total as f64 / 1e15;
                0.15 + 0.1 * staking_ratio.min(1.0)
            }

            PredictionDomain::StakingEconomics => {
                // APY target: 5-15%
                let validator_factor = (100.0 / context.validator_count as f64).min(1.0);
                0.05 + 0.10 * validator_factor
            }

            PredictionDomain::SecurityAnalysis => {
                // Security score: based on hashrate and validators
                let hashrate_score = (context.hashrate.ln() / 50.0).clamp(0.0, 0.5);
                let validator_score = (context.validator_count as f64 / 200.0).min(0.5);
                hashrate_score + validator_score
            }

            _ => 0.5, // Default neutral prediction
        }
    }
}

/// Fallback health metrics
#[derive(Clone, Debug, Default)]
struct FallbackHealth {
    twma_uses: u64,
    heuristic_uses: u64,
    last_use: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twma_fallback() {
        let mut twma = TWMAFallback::new(10);

        twma.record(PredictionDomain::FeeForecasting, 1.0);
        twma.record(PredictionDomain::FeeForecasting, 2.0);
        twma.record(PredictionDomain::FeeForecasting, 3.0);

        let result = twma.calculate(PredictionDomain::FeeForecasting);
        assert!(result.is_some());

        // Should be close to recent values (more weight)
        let value = result.unwrap();
        assert!(value > 1.5 && value < 3.5);
    }

    #[test]
    fn test_heuristic_fallback() {
        let heuristic = HeuristicFallback::new();

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

        let fee = heuristic.calculate(PredictionDomain::FeeForecasting, &context);
        assert!(fee > 0.0);

        let security = heuristic.calculate(PredictionDomain::SecurityAnalysis, &context);
        assert!(security >= 0.0 && security <= 1.0);
    }

    #[tokio::test]
    async fn test_fallback_system() {
        let fallback = OracleFallbackSystem::new();

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

        let prediction = fallback
            .get_prediction(PredictionDomain::FeeForecasting, &context)
            .await
            .unwrap();

        assert!(prediction.confidence < 0.5); // Low confidence for fallback
        assert!(matches!(prediction.source, PredictionSource::Fallback | PredictionSource::TWMA));
    }
}
